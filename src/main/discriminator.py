import torch
import torch.nn as nn
import torch.nn.functional as F
from main.config import Config

# L2 正则化在 PyTorch 中由优化器通过 weight_decay 参数处理。

class CommonPoseDiscriminator(nn.Module):
    def __init__(self):
        super(CommonPoseDiscriminator, self).__init__()
        self.config = Config()
        self.conv_2d_one = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=1)
        self.conv_2d_two = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)

    def forward(self, inputs):
        x = inputs.unsqueeze(2) # Corresponds to tf.expand_dims(inputs, 2)
        x = x.permute(0, 3, 1, 2) # Change to channels_first format
        x = self.conv_2d_one(x)
        x = F.relu(x)
        x = self.conv_2d_two(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1) # Change back to channels_last-like format
        return x

class SingleJointDiscriminator(nn.Module):
    def __init__(self):
        super(SingleJointDiscriminator, self).__init__()
        self.config = Config()
        self.joint_discriminators = nn.ModuleList()
        for i in range(self.config.NUM_JOINTS):
            self.joint_discriminators.append(nn.Linear(32, 1))

    def forward(self, inputs):
        single_joint_outputs = []
        for i in range(self.config.NUM_JOINTS):
            single_joint_outputs.append(self.joint_discriminators[i](inputs[:, i, :, :]))
        output = torch.squeeze(torch.stack(single_joint_outputs, 1))
        return output

class FullPoseDiscriminator(nn.Module):
    def __init__(self):
        super(FullPoseDiscriminator, self).__init__()
        self.config = Config()
        self.flatten = nn.Flatten()
        self.fc_one = nn.Linear(self.config.NUM_JOINTS * 32, 1024)
        self.fc_two = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, 1)

    def forward(self, inputs):
        x = self.flatten(inputs)
        x = self.fc_one(x)
        x = F.relu(x)
        x = self.fc_two(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x

class ShapeDiscriminator(nn.Module):
    def __init__(self):
        super(ShapeDiscriminator, self).__init__()
        self.config = Config()
        self.fc_one = nn.Linear(self.config.NUM_SHAPE_PARAMS, 10)
        self.fc_two = nn.Linear(10, 5)
        self.fc_out = nn.Linear(5, 1)

    def forward(self, inputs):
        x = self.fc_one(inputs)
        x = F.relu(x)
        x = self.fc_two(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.config = Config()
        self.common_pose_discriminator = CommonPoseDiscriminator()
        self.single_joint_discriminator = SingleJointDiscriminator()
        self.full_pose_discriminator = FullPoseDiscriminator()
        self.shape_discriminator = ShapeDiscriminator()

    def forward(self, inputs):
        poses = inputs[:, :self.config.NUM_JOINTS * 9]
        shapes = inputs[:, -self.config.NUM_SHAPE_PARAMS:]
        poses = poses.view(-1, self.config.NUM_JOINTS, 9)
        common_pose_features = self.common_pose_discriminator(poses)
        single_joint_outputs = self.single_joint_discriminator(common_pose_features)
        full_pose_outputs = self.full_pose_discriminator(common_pose_features)
        shape_outputs = self.shape_discriminator(shapes)
        return torch.cat((single_joint_outputs, full_pose_outputs, shape_outputs), 1)