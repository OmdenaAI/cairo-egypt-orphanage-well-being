import cv2
import numpy as np
import os
from tqdm import tqdm


class VideoEditer:
    def __init__(self, cfg):
        self.yolo_out= cfg['results_path']
        self.clip_duration= cfg['duration']
        self.fix_duration= cfg['fix_duration']

    def __call__(self, vid_path, save_path):

        video = cv2.VideoCapture(vid_path)
        frame_cnt= int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data= []
        with open(self.yolo_out, 'r') as f:
            for line in f:
                data.append(line.strip().split())

        # data need to be float then int, idk why :D
        data= np.array(data, float)
        data= np.array(data, int)
        assert frame_cnt == data[-1][-1], "Value Error, mismatch in results frames and original video frames"

        frame_col= data[:, -1]
        _, indices = np.unique(frame_col, return_index=True)
        subarrays = np.split(data, indices[1:])

        bboxes= []

        # iterate over the subarrays of the video to extract clips of the desired length
        for start_frame in range(0, frame_cnt, self.clip_duration * fps):
            # each clip_meta has clip_duration * fps frames matrix [list of detected objects]
            clip_meta= subarrays[start_frame:start_frame + self.clip_duration * fps]
            
            persons= []
            
            # iterate over each frame collection
            # if we sure that each frame contains at least 1 person
            # we can slice the first element without looping 
            for clip_frame in clip_meta:
                # here we interseted in persons only 
                persons_data= clip_frame[clip_frame[:,4] == 0]

                # no persons detected in the current frame
                if len(persons_data) == 0:
                    continue

                # getting each person in the frame to have his own video
                # here we're interested with detection happening in the first frame
                # and we track each id and crop based on the bbox
                unique_ids= persons_data[:, 5].tolist()
                persons.extend(unique_ids)
                start= persons_data
                break

                
            for id in persons:
                bbox_start= start[start[:, 5] == id][0]
                try:
                    bbox_end= self.get_bbox_last(clip_meta, id)
                except:
                    print(f'The ID: {id} has no other apperance in the next: {self.clip_duration} skipping...')
                    continue
                top_s, left_s, bottom_s, right_s, _, _, s_frame= bbox_start
                top_e, left_e, bottom_e, right_e, _, _, e_frame= bbox_end

                new_top, new_left, new_bottom, new_right= min(top_s, top_e), min(left_s, left_e), max(bottom_s, bottom_e), max(right_s, right_e)
                bboxes.append([new_top, new_left, new_bottom, new_right, id, s_frame, e_frame])

        for i, bbox in tqdm(enumerate(bboxes), unit= 'clips'):
            if self.fix_duration:
                if bbox[6] - bbox[5] != self.clip_duration * fps - 1:
                    continue
            video.set(cv2.CAP_PROP_POS_FRAMES, bbox[5])
            output_filename = f'output_{i}_of_{bbox[4]}_{bbox[5]}_{bbox[6]}.mp4'
            output_path = os.path.join(save_path, output_filename)
            output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (280, 360))

            for _ in range(bbox[5], bbox[6]):
                ret, frame = video.read()
                new_f= frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                new_f= cv2.resize(new_f, (280, 360)) 
                if not ret:
                    break
                output.write(new_f)

            output.release()

        video.release()


    def get_bbox_last(self, subarrays, id):
        for subarray in reversed(subarrays):
            for row in reversed(subarray):
                if row[5] == id:
                    return row
        return None 



if __name__ == "__main__":
    config= {'results_path': 'path/to/results/of/yolo', 'duration': 5, 'fix_duration': True}
    VE= VideoEditer(config)

    VE('original/clip/path', 'output/path')

                

                




        

