import h5py
import numpy as np
import torch
import torch.nn.functional as F
from main.config import Config

def batch_compute_similarity_transform(real_kp3d, pred_kp3d):
    real_kp3d = real_kp3d.transpose(1, 2)
    pred_kp3d = pred_kp3d.transpose(1, 2)
    
    mean_real = real_kp3d.mean(dim=2, keepdim=True)
    mean_pred = pred_kp3d.mean(dim=2, keepdim=True)
    
    centered_real = real_kp3d - mean_real
    centered_pred = pred_kp3d - mean_pred
    
    variance = (centered_pred ** 2).sum(dim=(-2, -1), keepdim=True)
    
    K = torch.matmul(centered_pred, centered_real.transpose(1, 2))
    
    u, s, v = torch.svd(K.cpu()) # SVD on CPU for speed
    u, s, v = u.to(K.device), s.to(K.device), v.to(K.device)
    
    det = torch.det(torch.matmul(u, v.transpose(1, 2))).sign()
    det = det.unsqueeze(-1).unsqueeze(-1)
    
    identity = torch.eye(u.shape[1], device=u.device).unsqueeze(0).repeat(u.shape[0], 1, 1)
    identity = identity * det
    
    R = torch.matmul(v, torch.matmul(identity, u.transpose(1, 2)))
    
    trace = torch.einsum('bii->b', torch.matmul(R, K)).unsqueeze(-1).unsqueeze(-1)
    scale = trace / variance
    
    trans = mean_real - scale * torch.matmul(R, mean_pred)
    
    aligned_kp3d = scale * torch.matmul(R, pred_kp3d) + trans
    
    return aligned_kp3d.transpose(1, 2)

def batch_align_by_pelvis(kp3d):
    left_id, right_id = 3, 2
    pelvis = (kp3d[:, left_id, :] + kp3d[:, right_id, :]) / 2.0
    return kp3d - pelvis.unsqueeze(1)

def batch_orthographic_projection(kp3d, camera):
    camera = camera.view(-1, 1, 3)
    kp_trans = kp3d[:, :, :2] + camera[:, :, 1:]
    shape = kp_trans.shape
    kp_trans = kp_trans.view(shape[0], -1)
    kp2d = camera[:, :, 0] * kp_trans
    return kp2d.view(shape)

def batch_skew_symmetric(vector):
    config = Config()
    batch_size = vector.shape[0]
    num_joints = config.NUM_JOINTS_GLOBAL
    
    zeros = torch.zeros(batch_size, num_joints, 1, device=vector.device)
    
    skew_sym = torch.stack(
        [zeros, -vector[:, :, 2:3], vector[:, :, 1:2],
         vector[:, :, 2:3], zeros, -vector[:, :, 0:1],
         -vector[:, :, 1:2], vector[:, :, 0:1], zeros],
        dim=-1)
    
    return skew_sym.view(batch_size, num_joints, 3, 3)

def batch_rodrigues(theta):
    config = Config()
    batch_size = theta.shape[0]
    num_joints = config.NUM_JOINTS_GLOBAL
    
    theta = theta.view(batch_size, num_joints, 3)
    
    angle = torch.norm(theta + 1e-8, dim=2, keepdim=True)
    axis = theta / angle
    
    skew_symm = batch_skew_symmetric(axis)
    
    cos_angle = torch.cos(angle).unsqueeze(-1)
    sin_angle = torch.sin(angle).unsqueeze(-1)
    
    axis = axis.unsqueeze(-1)
    outer = torch.matmul(axis, axis.transpose(2, 3))
    
    identity = torch.eye(3, device=theta.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_joints, 1, 1)
    
    rot_mat = identity * cos_angle + (1 - cos_angle) * outer + sin_angle * skew_symm
    rot_mat = rot_mat.view(batch_size, num_joints, 9)
    return rot_mat

def batch_global_rigid_transformation(rot_mat, joints, ancestors, rotate_base=False):
    config = Config()
    batch_size = rot_mat.shape[0]
    num_joints = config.NUM_JOINTS_GLOBAL
    
    if rotate_base:
        rot_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=rot_mat.device)
        rot_x = rot_x.unsqueeze(0).repeat(batch_size, 1, 1)
        root_rotation = torch.matmul(rot_mat[:, 0, :, :], rot_x)
    else:
        root_rotation = rot_mat[:, 0, :, :]
        
    def create_global_rot_for(_rotation, _joint):
        _rot_homo = F.pad(_rotation, (0, 0, 0, 1, 0, 0))
        _joint_homo = torch.cat([_joint, torch.ones(batch_size, 1, 1, device=_joint.device)], dim=1)
        _joint_world_trans = torch.cat([_rot_homo, _joint_homo], dim=2)
        return _joint_world_trans

    joints = joints.unsqueeze(-1)
    root_trans = create_global_rot_for(root_rotation, joints[:, 0])
    
    results = [root_trans]
    # In PyTorch, it's better to build the list first and then stack.
    # The direct append and index approach from TF is inefficient.
    # We will pre-calculate all transformations.
    
    # This loop is complex to translate directly due to dependencies.
    # A more PyTorch-idiomatic way would involve a batched tree traversal,
    # which is beyond a direct line-by-line translation.
    # The following is a functional, but potentially slow, translation.
    
    # Pre-allocate tensor for results
    results_tensor = torch.zeros(batch_size, num_joints, 4, 4, device=rot_mat.device)
    results_tensor[:, 0] = root_trans
    
    for i in range(1, ancestors.shape[0]):
        parent_idx = ancestors[i]
        joint_rel = joints[:, i] - joints[:, parent_idx]
        joint_glob_rot = create_global_rot_for(rot_mat[:, i], joint_rel)
        parent_trans = results_tensor[:, parent_idx]
        results_tensor[:, i] = torch.matmul(parent_trans, joint_glob_rot)
        
    new_joints = results_tensor[:, :, :3, 3]

    zeros = torch.zeros(batch_size, num_joints, 1, 1, device=joints.device)
    rest_pose = torch.cat([joints, zeros], dim=2)
    init_bone = torch.matmul(results_tensor, rest_pose)
    init_bone = F.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0)) # Pad last dimension
    rel_joints = results_tensor - init_bone
    
    return new_joints, rel_joints

def load_mean_theta():
    config = Config()
    with h5py.File(config.SMPL_MEAN_THETA_PATH, 'r') as f:
        mean_pose = f['pose'][:]
        mean_shape = f['shape'][:]
    
    mean = np.zeros((1, config.NUM_SMPL_PARAMS))
    
    mean_pose[:3] = 0.
    mean_pose[0] = np.pi
    
    mean[0, 0] = 0.9
    mean[:, config.NUM_CAMERA_PARAMS:] = np.hstack((mean_pose, mean_shape))
    
    return torch.tensor(mean, dtype=torch.float32)