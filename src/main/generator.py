import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from main import model_util
from main.config import Config
from main.smpl import Smpl


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.config = Config()

        self.mean_theta = nn.Parameter(torch.from_numpy(model_util.load_mean_theta()))

        self.fc_one = nn.Linear(2048 + 85, 1024)
        self.dropout_one = nn.Dropout(0.5)
        self.fc_two = nn.Linear(1024, 1024)
        self.dropout_two = nn.Dropout(0.5)
        self.fc_out = nn.Linear(1024, 85)
        
        # Initialize fc_out weights
        nn.init.uniform_(self.fc_out.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        batch_theta = self.mean_theta.expand(batch_size, -1)
        
        thetas = []
        for i in range(self.config.ITERATIONS):
            total_inputs = torch.cat([inputs, batch_theta], dim=1)
            batch_theta = batch_theta + self._fc_blocks(total_inputs)
            thetas.append(batch_theta)
            
        return torch.stack(thetas, dim=0)

    def _fc_blocks(self, inputs):
        x = self.fc_one(inputs)
        x = F.relu(x)
        x = self.dropout_one(x)
        x = self.fc_two(x)
        x = F.relu(x)
        x = self.dropout_two(x)
        x = self.fc_out(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.config = Config()

        self.enc_shape = self.config.ENCODER_INPUT_SHAPE
        
        # Load pretrained ResNet50 and remove the final fully connected layer
        resnet50_pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50 = nn.Sequential(*list(resnet50_pretrained.children())[:-1])
        self.flatten = nn.Flatten()
        
        # Note: The _set_resnet_arg_scope logic for custom initialization and regularization
        # is handled differently in PyTorch. Regularization is applied in the optimizer.
        # Custom initializations would typically be done in a separate function
        # that iterates through model.modules(). For direct translation, we omit this.

        self.regressor = Regressor()
        self.smpl = Smpl()

    def forward(self, inputs):
        features = self.resnet50(inputs)
        features = self.flatten(features)
        thetas = self.regressor(features)
        
        outputs = []
        for i in range(self.config.ITERATIONS):
            theta = thetas[i, :, :] # Correct indexing for stacked tensor
            output_tuple = self._compute_output(theta)
            outputs.append(output_tuple)
            
        return outputs

    def _compute_output(self, theta):
        cams = theta[:, :self.config.NUM_CAMERA_PARAMS]
        pose_and_shape = theta[:, self.config.NUM_CAMERA_PARAMS:]
        vertices, joints_3d, rotations = self.smpl(pose_and_shape)
        joints_2d = model_util.batch_orthographic_projection(joints_3d, cams)
        shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]
        
        return (vertices, joints_2d, joints_3d, rotations, shapes, cams)