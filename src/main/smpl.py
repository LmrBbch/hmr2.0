import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main.config import Config
from main.model_util import batch_rodrigues, batch_global_rigid_transformation

class Smpl(nn.Module):
    def __init__(self):
        super(Smpl, self).__init__()
        self.config = Config()
        if self.config.JOINT_TYPE not in ['cocoplus', 'lsp', 'custom']:
            raise Exception('unknown joint type: {}, it must be either cocoplus or lsp'.format(self.config.JOINT_TYPE))

        with open(self.config.SMPL_MODEL_PATH, 'rb') as f:
            # In PyTorch, we load data into tensors and register them as non-trainable buffers
            model_data = pickle.load(f, encoding='latin1')

        def to_tensor(data, dtype=torch.float32):
            return torch.tensor(data, dtype=dtype)
        
        # Buffers are part of the model's state but are not parameters to be trained.
        self.register_buffer('vertices_template', to_tensor(model_data['v_template']))
        
        shapes = to_tensor(model_data['shapedirs'])
        self.num_betas = shapes.shape[-1]
        self.register_buffer('shapes', shapes.view(-1, self.num_betas).T)
        
        self.register_buffer('smpl_joint_regressor', to_tensor(model_data['J_regressor'].T))
        
        pose = to_tensor(model_data['posedirs'])
        self.register_buffer('pose', pose.view(-1, pose.shape[-1]).T)
        
        self.register_buffer('lbs_weights', to_tensor(model_data['weights']))
        
        self.register_buffer('faces', to_tensor(model_data['f'].astype(np.int64)))
        
        joint_regressor = model_data['cocoplus_regressor']
        if self.config.JOINT_TYPE == 'custom':
            if len(self.config.CUSTOM_REGRESSOR_IDX) > 0:
                for index, file_name in self.config.CUSTOM_REGRESSOR_IDX.items():
                    file = join(self.config.CUSTOM_REGRESSOR_PATH, file_name)
                    regressor = np.load(file)
                    joint_regressor = np.insert(joint_regressor, index, np.squeeze(regressor), 0)
        else:
            if self.config.INITIALIZE_CUSTOM_REGRESSOR:
                joint_regressor_plus_np = np.copy(joint_regressor)
                for index, file_name in self.config.CUSTOM_REGRESSOR_IDX.items():
                    file = join(self.config.CUSTOM_REGRESSOR_PATH, file_name)
                    regressor = np.load(file).astype(np.float32)
                    joint_regressor_plus_np = np.insert(joint_regressor_plus_np, index, np.squeeze(regressor), 0)
                self.register_buffer('joint_regressor_plus', to_tensor(joint_regressor_plus_np.T))

        if self.config.JOINT_TYPE == 'lsp':
            joint_regressor = joint_regressor[:, :14]
        
        self.register_buffer('joint_regressor', to_tensor(joint_regressor.T))
        
        self.ancestors = model_data['kintree_table'][0].astype(np.int32)
        self.register_buffer('identity', torch.eye(3))
        self.joint_transformed = None

    def forward(self, inputs):
        _batch_size = inputs.shape[0]
        _pose = inputs[:, :self.config.NUM_POSE_PARAMS]
        _shape = inputs[:, -self.config.NUM_SHAPE_PARAMS:]
        _reshape_shape = (_batch_size, self.vertices_template.shape[0], self.vertices_template.shape[1])
        
        v_shaped = torch.matmul(_shape, self.shapes).view(_reshape_shape) + self.vertices_template
        
        v_joints = self.compute_joints(v_shaped, self.smpl_joint_regressor)
        
        rotations = batch_rodrigues(_pose).view(_batch_size, self.config.NUM_JOINTS_GLOBAL, 3, 3)
        
        pose_feature = (rotations[:, 1:, :, :] - self.identity).view(_batch_size, -1)
        
        v_posed = torch.matmul(pose_feature, self.pose).view(_reshape_shape) + v_shaped
        
        self.joint_transformed, rel_joints = batch_global_rigid_transformation(rotations, v_joints, self.ancestors)
        
        weights = self.lbs_weights.unsqueeze(0).expand(_batch_size, -1, -1)
        
        rel_joints = rel_joints.view(_batch_size, self.config.NUM_JOINTS_GLOBAL, 16)
        weighted_joints = torch.matmul(weights, rel_joints).view(_batch_size, -1, 4, 4)
        
        ones = torch.ones(_batch_size, v_posed.shape[1], 1, device=inputs.device)
        v_posed_homo = torch.cat([v_posed, ones], dim=2).unsqueeze(-1)
        
        v_posed_homo = torch.matmul(weighted_joints, v_posed_homo)
        
        vertices = v_posed_homo[:, :, :3, 0]
        
        if self.config.JOINT_TYPE != 'custom' and self.config.INITIALIZE_CUSTOM_REGRESSOR:
            joints = self.compute_joints(vertices, self.joint_regressor_plus)
        else:
            joints = self.compute_joints(vertices, self.joint_regressor)
            
        return vertices, joints, rotations

    def compute_joints(self, vertices, regressor):
        joint_x = torch.matmul(vertices[:, :, 0], regressor)
        joint_y = torch.matmul(vertices[:, :, 1], regressor)
        joint_z = torch.matmul(vertices[:, :, 2], regressor)
        return torch.stack([joint_x, joint_y, joint_z], dim=2)

    def get_faces(self):
        return self.faces

    def save_obj(self, _vertices, file_name):
        file = './{}.obj'.format(file_name)
        with open(file, 'w') as fp:
            for v in _vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            
            for f in self.faces.cpu().numpy():
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))