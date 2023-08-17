import torch
import torch.nn as nn
import torch.nn.functional as F

# this file mainly used for fine tuning i3d if needed (the frozen BN)

class FrozenBN(nn.Module):
	"""
	 Batch normalization is a technique commonly used in deep learning models to improve training stability and accelerate convergence.
	 In the case of this module, it allows applying pre-defined batch normalization parameters to input tensors without updating them during the training process
	"""
	def __init__(self, num_channels, momentum=0.1, eps=1e-5):
		"""
		num_channels: The number of channels in the input tensor.
		momentum: The momentum factor for updating the running mean and variance.
		eps: The epsilon value added to the variance to avoid division by zero.
		"""
		super(FrozenBN, self).__init__()
		self.num_channels = num_channels
		self.momentum = momentum
		self.eps = eps
		self.params_set = False

	# set_params method sets the batch normalization parameters required for the forward pass
	def set_params(self, scale, bias, running_mean, running_var):
		"""
		scale: The learnable scale parameter for each channel.
		bias: The learnable bias parameter for each channel.
		running_mean: The running mean of the batch normalization
		running_var: The running variance of the batch normalization

		These parameters are registered as buffers in the module, ensuring they are saved and loaded together with the model
		"""
		self.register_buffer('scale', scale)
		self.register_buffer('bias', bias)
		self.register_buffer('running_mean', running_mean)
		self.register_buffer('running_var', running_var)
		self.params_set = True

	# The forward method performs the forward pass of the module. It takes an input tensor x and applies batch normalization using the pre-defined parameters
	def forward(self, x):
		assert self.params_set, 'model.set_params(...) must be called before the forward pass'
		return torch.batch_norm(x, self.scale, self.bias, self.running_mean, self.running_var, False, self.momentum, self.eps, torch.backends.cudnn.enabled)

	# The __repr__ method returns a string representation of the module, indicating the number of channels it operates on.
	def __repr__(self):
		return 'FrozenBN(%d)'%self.num_channels

def freeze_bn(m, name):
	"""
	recursively traverses a PyTorch module m and replaces all instances of torch.nn.BatchNorm3d with FrozenBN layers
	This function is used to freeze the batch normalization layers in a model by replacing them with pre-defined batch normalization parameters.

	m: The PyTorch module to freeze the batch normalization layers in
	name: The name of the current module.
	"""
	for attr_str in dir(m):
		target_attr = getattr(m, attr_str)
		# If an attribute is a BatchNorm3d layer, it creates a FrozenBN layer (frozen_bn) with the same number of features,
		# momentum, and epsilon as the original batch normalization layer
		if type(target_attr) == torch.nn.BatchNorm3d:
			# The parameters of the FrozenBN layer are set using the pre-defined parameters of the original batch normalization layer
			frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
			frozen_bn.set_params(target_attr.weight.data, target_attr.bias.data, target_attr.running_mean, target_attr.running_var)

	 		# The setattr function is used to replace the original BatchNorm3d layer with the FrozenBN layer in the current module.
			setattr(m, attr_str, frozen_bn)

	# recursively calls itself on each child module of the current module using the named_children function to freeze the batch normalization layers in the child modules as well.
	for n, ch in m.named_children():
		freeze_bn(ch, n)
		

class Bottleneck(nn.Module):
	"""
	building block for the ResNet architecture in the context of 3D convolutional neural networks.
	It consists of three convolutional layers with batch normalization and ReLU activation, along with
	optional downsampling and non-local operations
	"""
	expansion = 4

	def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
		super(Bottleneck, self).__init__()
		# The first convolutional layer and batch normalization layer, which reduce the number of input channels to
		# planes using a 1D temporal convolution with kernel size (1 + temp_conv * 2, 1, 1) and stride (temp_stride, 1, 1)
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
		self.bn1 = nn.BatchNorm3d(planes)

		# The second convolutional layer and batch normalization layer, which perform a 3D spatial convolution with
		# kernel size (1, 3, 3) and stride (1, stride, stride). This layer operates on the output of the first layer
		self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
		self.bn2 = nn.BatchNorm3d(planes)

		# The third convolutional layer and batch normalization layer, which increase the number of channels to planes * 4 using a 1x1x1 convolution.
		self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * 4)

		# ReLU activation function applied after each convolutional layer.
		self.relu = nn.ReLU(inplace=True)

		# Optional downsampling operation applied to the input x if the number of input channels is different from the number of output channels.
		# This operation helps match the dimensions of the residual connection.
		self.downsample = downsample
		self.stride = stride

		outplanes = planes * 4

		# An optional non-local block (NonLocalBlock) that can be added after the bottleneck block to capture long-range dependencies in the input data
		self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		if self.nl is not None:
			out = self.nl(out)

		return out
	
class NonLocalBlock(nn.Module):
	"""
	 class represents a non-local block module that captures long-range dependencies in the input feature maps
	"""
	def __init__(self, dim_in, dim_out, dim_inner):
		super(NonLocalBlock, self).__init__()

		self.dim_in = dim_in
		self.dim_inner = dim_inner
		self.dim_out = dim_out

		# self.theta, self.phi, self.g: Convolutional layers used to transform the input tensor into three feature representations: theta, phi, and g
		self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		# Max pooling operation applied to the input tensor to reduce its spatial dimensions
		self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

		# Convolutional layer used to transform the output tensor of the non-local block.
		self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		# Batch normalization layer applied to the output tensor
		self.bn = nn.BatchNorm3d(dim_out)

	def forward(self, x):
		residual = x

		batch_size = x.shape[0]
		mp = self.maxpool(x)
		theta = self.theta(x)
		phi = self.phi(mp)
		g = self.g(mp)

		theta_shape_5d = theta.shape
		theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner, -1)

		theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
		theta_phi_sc = theta_phi * (self.dim_inner**-.5)
		p = F.softmax(theta_phi_sc, dim=-1)

		t = torch.bmm(g, p.transpose(1, 2))
		t = t.view(theta_shape_5d)

		out = self.out(t)
		out = self.bn(out)

		out = out + residual
		return out
