import torch
import torch.nn as nn
from torch.autograd import Variable

from I3D_utils import Bottleneck
from PIL import Image
import os
import numpy as np
from pathlib import Path

import shutil
import time
import ffmpeg

from natsort import natsorted


def I3D(cfg, load_path=None):
    if load_path:
        try:
            print('[*] Attempting to load model from:', load_path)
            model = _I3D(cfg)
            model.load_state_dict(torch.load(load_path))
        except: 
            print('[*] Model does not exist or is corrupted. Creating new model...')
            return _I3D(cfg)

        # check whether `model` is an _I3D instance
        if model.__class__.__name__ == '_I3D':
            return model
        else:
            raise ValueError('The loaded tensor is not an instance of _I3D.')
    else:
        print('[*] Creating model...')
        return _I3D(cfg)

class _I3D(nn.Module):
    """
    class represents a modified version of the 3D ResNet-50 model, also known as I3D, which is commonly used for video classification tasks.
    """

    # block and layers: These parameters determine the type of residual block used (block) and the number of blocks in each stage of the network (layers)
    def __init__(self, cfg):
        # This variable keeps track of the number of input channels to the next residual block.
        self.inplanes = 64
        super(_I3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if cfg['use_nl'] else 1000
        # These are the four stages of the network, each consisting of multiple residual blocks.
        # The _make_layer method is called to create each stage. The number of blocks, the number of output channels for each block,
        # and the temporal convolution and stride configurations are specified for each stage.
        self.layer1 = self._make_layer(cfg['block'], 64, cfg['layers'][0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(cfg['block'], 128, cfg['layers'][1], stride=2, temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(cfg['block'], 256, cfg['layers'][2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(cfg['block'], 512, cfg['layers'][3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * cfg['block'].expansion, cfg['num_classes'])
        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        """
        The _make_layer method is responsible for creating a stage of residual blocks. It takes the block type,
        the number of output channels for each block (planes), the number of blocks (blocks), the stride value, temporal
        convolution configurations (temp_conv), temporal stride configurations (temp_stride), and a nonlocal_mod parameter
        that determines whether to use non-local blocks in certain blocks of the stage.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i], i%nonlocal_mod==nonlocal_mod-1))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        """
        The forward_single method defines the forward pass of the network for a single video clip
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, batch):
        """
        The forward method handles the forward pass for a batch of video clips.
        """
        if batch['frames'].dim() == 5:
            feat = self.forward_single(batch['frames'])
        return feat
    
    


class Modeling_I3D(nn.Module):
    def __init__(self, i3d_cfg, pretrainedpath):
        super(Modeling_I3D, self).__init__()
        self.i3d= I3D(i3d_cfg, pretrainedpath)


    def _load_frame(self, frame_file):
        """
        load and preprocess a single frame from an image file
        """
        data = Image.open(frame_file)
        data = data.resize((340, 256), Image.ANTIALIAS)
        data = np.array(data)
        data = data.astype(float)
        data = (data * 2 / 255) - 1
        assert(data.max()<=1.0)
        assert(data.min()>=-1.0)
        return data
    
    def _load_rgb_batch(self, frames_dir, rgb_files, frame_indices):
        batch_data = np.zeros(frame_indices.shape + (256,340,3))
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                batch_data[i,j,:,:,:] = self._load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
        return batch_data
    

    def _oversample_data(self, data):
        """
        perform data augmentation by oversampling the given data
        """
        # Create a flipped version of the input data by reversing the order of the last dimension
        data_flip = np.array(data[:,:,:,::-1,:])

        # Extract different spatial regions from the original and flipped data using slicing
        # Select the first 224x224 region from the top-left corner of each frame
        data_1 = np.array(data[:, :, :224, :224, :])
        # Select the first 224x224 region from the top-right corner of each frame
        data_2 = np.array(data[:, :, :224, -224:, :])
        # Select a 224x224 region starting from (16, 58) to (240, 282) of each frame
        data_3 = np.array(data[:, :, 16:240, 58:282, :])
        # Select the last 224x224 region from the bottom-left corner of each frame
        data_4 = np.array(data[:, :, -224:, :224, :])
        # Select the last 224x224 region from the bottom-right corner of each frame
        data_5 = np.array(data[:, :, -224:, -224:, :])

        # Repeat the same extraction process for the flipped data to obtain the corresponding flipped spatial regions
        data_f_1 = np.array(data_flip[:, :, :224, :224, :])
        data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
        data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
        data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
        data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

        return [data_1, data_2, data_3, data_4, data_5,
                data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
    
    def _forward_batch(self, b_data, i3d):
        """
        This function converts the numpy array to a PyTorch tensor, performs the forward pass through the i3d model,
        and returns the extracted features as a numpy array
        """
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
        with torch.no_grad():
            b_data = Variable(b_data.cuda()).float()
            inp = {'frames': b_data}
            features = i3d(inp)
        return features.cpu().numpy()

    def _run(self, i3d, frequency, frames_dir, batch_size, sample_mode):
        """
        function that takes an I3D model, a frequency, a directory of RGB frames, a batch size,
        and a sample mode, and returns a tensor of features extracted from the RGB frames
        :i3d: the I3D model
        :frequency: the frequency at which frames are sampled to form video clips
        :frames_dir: directory containing the frames
        :batch_size: the batch size for processing frames
        :sample_mode: the sampling strategy, either 'oversample' or 'center_crop'
        """
        assert(sample_mode in ['oversample', 'center_crop'])
        print("batchsize", batch_size)

        # the number of consecutive frames that are considered as a single video clip or chunk
        # It represents the length of each video clip that will be processed by the I3D model
        # The frames are divided into chunks to create meaningful sequences for video analysis
        # Adjusting the chunk_size can affect the temporal context captured by the model.
        # A larger chunk_size can capture longer-term dependencies but may require more computational resources,
        # while a smaller chunk_size may capture shorter-term dependencies but can be more computationally efficient.
        chunk_size = 16

        # The RGB files in the frames_dir directory are sorted using natsorted to ensure they are in the correct order.
        rgb_files = natsorted([i for i in os.listdir(frames_dir)])

        # total number of frames
        frame_cnt = len(rgb_files)

        assert(frame_cnt > chunk_size)
        # The clipped_length is computed by subtracting the chunk size from the total number of frames
        # This value determines the starting point of the last chunk
        clipped_length = frame_cnt - chunk_size
        clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk

        # The frame indices are generated based on the clipped_length and frequency.
        # Each frame index represents a chunk of frames that will be used as a video clip
        # The number of chunks is determined by dividing the clipped_length by the frequency and adding 1.
        # This is then used to calculate the number of batches needed for processing
        frame_indices = [] # Frames to chunks

        for i in range(clipped_length // frequency + 1):
            frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])

        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]
        batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
        # The frame indices are split into batches using np.array_split to evenly distribute the chunks among the batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        # If the sample_mode is set to 'oversample', data augmentation is performed using the _oversample_data function.
        # This function generates different variations of the frames, such as flipping and cropping, to create augmented samples.
        if sample_mode == 'oversample':
            full_features = [[] for i in range(10)]
        else:
            full_features = [[]]

        # For each batch, the _load_rgb_batch function is called to load the frames corresponding to the frame indices.
        # This function returns a numpy array of shape (batch_size, chunk_size, 256, 340, 3) containing the frames.
        for batch_id in range(batch_num):
            batch_data = self._load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])

            # If the sample_mode is 'oversample', the _oversample_data function is applied to the batch data to generate multiple variations of the frames.
            # The resulting data is stored in batch_data_ten_crop, which is a list of numpy arrays.
            if(sample_mode == 'oversample'):
                batch_data_ten_crop = self._oversample_data(batch_data)
                for i in range(10):
                    assert(batch_data_ten_crop[i].shape[-2]==224)
                    assert(batch_data_ten_crop[i].shape[-3]==224)
                    # The _forward_batch function is called for each variation of the frames.
                    temp = self._forward_batch(batch_data_ten_crop[i], i3d)
                    # The extracted features are appended to the corresponding sublist in full_features
                    full_features[i].append(temp)

            # If the sample_mode is 'center_crop', the batch data is cropped using the specified coordinates (16:240, 58:282).
            elif(sample_mode == 'center_crop'):
                batch_data = batch_data[:,:,16:240,58:282,:]
                assert(batch_data.shape[-2]==224)
                assert(batch_data.shape[-3]==224)
                # The _forward_batch function is called for each variation of the frames.
                temp = self._forward_batch(batch_data, i3d)
                # The extracted features are appended to the corresponding sublist in full_features
                full_features[0].append(temp)

        # After processing all batches, the extracted features in full_features are concatenated along the first axis to form a single numpy array.
        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)
        # The dimensions of the array are reshaped to remove redundant dimensions and transpose the axes
        full_features = full_features[:,:,:,0,0,0]
        full_features = np.array(full_features).transpose([1,0,2])
        # The resulting features are returned as a numpy array of shape
        # (num_features, num_samples, feature_size), where num_features is the number of variations (10 for 'oversample', 1 for 'center_crop'),
        # num_samples is the total number of video clips, and feature_size is the size of the extracted features.
        return full_features

    def _preprocess(self, x):
        return x.reshape(x.shape[0]*x.shape[1], x.shape[2])

    def generate(self, data_cfg):
        Path(data_cfg['outputpath']).mkdir(parents=True, exist_ok=True)
        temppath = data_cfg['outputpath']+ "/temp/"
        rootdir = Path(data_cfg['datasetpath'])
        videos = [str(f) for f in rootdir.glob('**/*.mp4')]

        self.i3d.cuda()
        self.i3d.train(False)

        for video in videos:
            _, tail= os.path.split(video)
            videoname = tail.split(".")[0]
            startime = time.time()
            print("Generating for {0}".format(video))
            Path(temppath).mkdir(parents=True, exist_ok=True)
            ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
            print("Preprocessing done..")
            features = self._run(self.i3d, data_cfg['frequency'], temppath, data_cfg['batch_size'], data_cfg['sample_mode'])
            new_features= self._preprocess(features)
            np.save(os.path.join(data_cfg['outputpath'], videoname), new_features)
            print("Obtained features of size: ", new_features.shape)
            shutil.rmtree(temppath)
            print("done in {0}.".format(time.time() - startime))


if __name__ == "__main__":
    # repository_root = Path.cwd().parent.parent
    # print(repository_root)
    data_cfg= {'datasetpath': 'list/of/clips/with/annotations', 
               'outputpath': 'path/to/output', 
               'frequency': 16, 'batch_size': 20, 'sample_mode': 'oversample'}
    model_cfg= {'num_classes': 400, 'use_nl': False, 'block': Bottleneck, 'layers': [3, 4, 6, 3]}
   
    model= Modeling_I3D(model_cfg, "path/to/kinetics")  # https://drive.google.com/file/d/1iJRL_sI88ojxM5vUHwYaXXDM9zgw6rhO/view?usp=sharing
    model.generate(data_cfg)
    

