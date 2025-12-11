import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# to make run from console for module import
sys.path.append(os.path.abspath(".."))

from main.config import Config
from main.dataset import Dataset # Assumes this returns PyTorch Datasets/DataLoaders
from main.discriminator import Discriminator # PyTorch version
from main.generator import Generator # PyTorch version
from main.model_util import batch_align_by_pelvis, batch_compute_similarity_transform, batch_rodrigues

class ExceptionHandlingIterator:
    def __init__(self, iterable):
        self._iter = iter(iterable)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration as e:
            raise e
        except Exception as e:
            print(e)
            return self.__next__()

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def reset(self):
        self.__init__()
    def result(self):
        return self.avg

class Model:
    def __init__(self, display_config=True):
        self.config = Config()
        self.config.save_config()
        if display_config:
            self.config.display()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._build_model()
        self._setup_summary()

    def _build_model(self):
        self.generator = Generator().to(self.device)
        self.generator_opt = optim.Adam(self.generator.parameters(), lr=self.config.GENERATOR_LEARNING_RATE)
        if not self.config.ENCODER_ONLY:
            self.discriminator = Discriminator().to(self.device)
            self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=self.config.DISCRIMINATOR_LEARNING_RATE)
        
        self.checkpoint_dir = self.config.LOG_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        latest_checkpoint = self._find_latest_checkpoint()
        if self.config.RESTORE_PATH:
            self._load_checkpoint(self.config.RESTORE_PATH)
        elif latest_checkpoint:
            self._load_checkpoint(latest_checkpoint)

    def _find_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth.tar')]
        if not checkpoints: return None
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator_opt.load_state_dict(checkpoint['generator_opt_state_dict'])
        if not self.config.ENCODER_ONLY:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.discriminator_opt.load_state_dict(checkpoint['discriminator_opt_state_dict'])

    def _save_checkpoint(self, epoch):
        state = {'epoch': epoch, 'generator_state_dict': self.generator.state_dict(), 'generator_opt_state_dict': self.generator_opt.state_dict()}
        if not self.config.ENCODER_ONLY:
            state['discriminator_state_dict'] = self.discriminator.state_dict()
            state['discriminator_opt_state_dict'] = self.discriminator_opt.state_dict()
        filename = os.path.join(self.checkpoint_dir, f"ckpt_{epoch}.pth.tar")
        torch.save(state, filename)

    def _setup_summary(self):
        self.summary_path = os.path.join(self.config.LOG_DIR, 'hmr2.0', '3D_{}'.format(self.config.USE_3D))
        self.summary_writer = SummaryWriter(self.summary_path)
        self.generator_loss_log = AverageMeter()
        self.kp2d_loss_log = AverageMeter()
        self.gen_disc_loss_log = AverageMeter()
        if self.config.USE_3D:
            self.kp3d_loss_log = AverageMeter()
            self.pose_shape_loss_log = AverageMeter()
        self.discriminator_loss_log = AverageMeter()
        self.disc_real_loss_log = AverageMeter()
        self.disc_fake_loss_log = AverageMeter()
        self.kp2d_mpjpe_log = AverageMeter()
        self.kp3d_mpjpe_log = AverageMeter()
        self.kp3d_mpjpe_aligned_log = AverageMeter()

    def train(self):
        dataset = Dataset()
        ds_train, ds_smpl = dataset.get_train(), dataset.get_smpl()
        ds_val = dataset.get_val()
        
        start_epoch = 1
        if self.config.RESTORE_EPOCH:
            start_epoch = self.config.RESTORE_EPOCH
            
        for epoch in range(start_epoch, self.config.EPOCHS + 1):
            start_time = time.time()
            dataset_train = ExceptionHandlingIterator(zip(ds_train, ds_smpl))
            
            for image_data, theta in tqdm(dataset_train, total=len(ds_train)):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._train_step(images.to(self.device), kp2d.to(self.device), kp3d.to(self.device), has3d.to(self.device), theta.to(self.device))
            self._log_train(epoch=epoch)

            for image_data in tqdm(ds_val, total=len(ds_val)):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._val_step(images.to(self.device), kp2d.to(self.device), kp3d.to(self.device), has3d.to(self.device))
            self._log_val(epoch=epoch)
            
            if epoch % 5 == 0:
                self._save_checkpoint(epoch)

        self.summary_writer.flush()
        self._save_checkpoint(self.config.EPOCHS + 1)
        self.summary_writer.close()

    def _train_step(self, images, kp2d, kp3d, has3d, theta):
        self.generator.train()
        self.discriminator.train()
        batch_size = images.shape[0]

        # --- Train Generator ---
        self.generator_opt.zero_grad()
        generator_outputs = self.generator(images)
        _, kp2d_pred, kp3d_pred, pose_pred, shape_pred, _ = generator_outputs[-1]
        
        vis = kp2d[:, :, 2].unsqueeze(-1)
        kp2d_loss = F.l1_loss(kp2d[:, :, :2], kp2d_pred, reduction='none') * vis
        kp2d_loss = kp2d_loss.sum() / vis.sum() * self.config.GENERATOR_2D_LOSS_WEIGHT
        
        generator_loss = kp2d_loss
        
        if self.config.USE_3D:
            has3d_w = has3d.unsqueeze(-1).float()
            kp3d_real = batch_align_by_pelvis(kp3d)
            kp3d_pred_aligned = batch_align_by_pelvis(kp3d_pred[:, :self.config.NUM_KP3D, :])
            kp3d_loss = F.mse_loss(kp3d_real.view(batch_size, -1), kp3d_pred_aligned.view(batch_size, -1), reduction='none') * has3d_w
            kp3d_loss = kp3d_loss.sum() / has3d_w.sum() * 0.5 * self.config.GENERATOR_3D_LOSS_WEIGHT
            
            pose_shape_pred = torch.cat([pose_pred.view(batch_size, -1), shape_pred.view(batch_size, -1)], 1)
            has_smpl = torch.zeros(batch_size, 1, device=self.device)
            pose_shape_real = torch.zeros_like(pose_shape_pred)
            ps_loss = F.mse_loss(pose_shape_real, pose_shape_pred, reduction='none') * has_smpl
            ps_loss = ps_loss.sum() / (has_smpl.sum() + 1e-9) * 0.5 * self.config.GENERATOR_3D_LOSS_WEIGHT
            generator_loss += kp3d_loss + ps_loss

        fake_disc_input = self.accumulate_fake_disc_input(generator_outputs)
        fake_disc_output = self.discriminator(fake_disc_input)
        gen_disc_loss = torch.mean((fake_disc_output - 1.0) ** 2) * self.config.DISCRIMINATOR_LOSS_WEIGHT
        generator_loss += gen_disc_loss
        
        generator_loss.backward()
        self.generator_opt.step()
        
        # --- Train Discriminator ---
        self.discriminator_opt.zero_grad()
        real_disc_input = self.accumulate_real_disc_input(theta)
        real_disc_output = self.discriminator(real_disc_input)
        disc_real_loss = torch.mean((real_disc_output - 1.0) ** 2)
        
        # Use detached fake input for discriminator training
        fake_disc_input_detached = self.accumulate_fake_disc_input(generator_outputs).detach()
        fake_disc_output_detached = self.discriminator(fake_disc_input_detached)
        disc_fake_loss = torch.mean(fake_disc_output_detached ** 2)

        discriminator_loss = (disc_real_loss + disc_fake_loss) * self.config.DISCRIMINATOR_LOSS_WEIGHT
        discriminator_loss.backward()
        self.discriminator_opt.step()

        # Logging
        n = batch_size
        self.generator_loss_log.update(generator_loss.item(), n)
        self.kp2d_loss_log.update(kp2d_loss.item(), n)
        self.gen_disc_loss_log.update(gen_disc_loss.item(), n)
        if self.config.USE_3D:
            self.kp3d_loss_log.update(kp3d_loss.item(), n)
            self.pose_shape_loss_log.update(ps_loss.item(), n)
        self.discriminator_loss_log.update(discriminator_loss.item(), n)
        self.disc_real_loss_log.update(disc_real_loss.item(), n)
        self.disc_fake_loss_log.update(disc_fake_loss.item(), n)

    def accumulate_fake_disc_input(self, generator_outputs):
        fake_poses = torch.stack([out[3] for out in generator_outputs])
        fake_shapes = torch.stack([out[4] for out in generator_outputs])
        fake_poses = fake_poses.view(-1, self.config.NUM_JOINTS_GLOBAL, 9)[:, 1:, :]
        fake_poses = fake_poses.view(-1, self.config.NUM_JOINTS * 9)
        fake_shapes = fake_shapes.view(-1, self.config.NUM_SHAPE_PARAMS)
        return torch.cat([fake_poses, fake_shapes], 1)

    def accumulate_real_disc_input(self, theta):
        real_poses = theta[:, :self.config.NUM_POSE_PARAMS]
        real_poses = batch_rodrigues(real_poses)[:, 1:, :]
        real_poses = real_poses.view(-1, self.config.NUM_JOINTS * 9)
        real_shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]
        return torch.cat([real_poses, real_shapes], 1)

    def _log_train(self, epoch):
        self.summary_writer.add_scalar('train/generator_loss', self.generator_loss_log.result(), epoch)
        self.summary_writer.add_scalar('train/kp2d_loss', self.kp2d_loss_log.result(), epoch)
        self.summary_writer.add_scalar('train/gen_disc_loss', self.gen_disc_loss_log.result(), epoch)
        if self.config.USE_3D:
            self.summary_writer.add_scalar('train/kp3d_loss', self.kp3d_loss_log.result(), epoch)
            self.summary_writer.add_scalar('train/pose_shape_loss', self.pose_shape_loss_log.result(), epoch)
        self.summary_writer.add_scalar('train/discriminator_loss', self.discriminator_loss_log.result(), epoch)
        self.summary_writer.add_scalar('train/disc_real_loss', self.disc_real_loss_log.result(), epoch)
        self.summary_writer.add_scalar('train/disc_fake_loss', self.disc_fake_loss_log.result(), epoch)
        
        self.generator_loss_log.reset()
        self.kp2d_loss_log.reset()
        self.gen_disc_loss_log.reset()
        if self.config.USE_3D:
            self.kp3d_loss_log.reset()
            self.pose_shape_loss_log.reset()
        self.discriminator_loss_log.reset()
        self.disc_real_loss_log.reset()
        self.disc_fake_loss_log.reset()

    def _val_step(self, images, kp2d, kp3d, has3d):
        self.generator.eval()
        with torch.no_grad():
            result = self.generator(images)
            _, kp2d_pred, kp3d_pred, _, _, _ = result[-1]
            vis = kp2d[:, :, 2].float()
            kp2d_norm = torch.norm(kp2d_pred[:, :self.config.NUM_KP2D, :] - kp2d[:, :, :2], dim=2) * vis
            kp2d_mpjpe = kp2d_norm.sum() / vis.sum()
            self.kp2d_mpjpe_log.update(kp2d_mpjpe.item(), vis.sum().item())
            
            if self.config.USE_3D and has3d.sum() > 0:
                kp3d_real = kp3d[has3d]
                kp3d_predict = kp3d_pred[has3d][:, :self.config.NUM_KP3D, :]
                kp3d_real = batch_align_by_pelvis(kp3d_real)
                kp3d_predict = batch_align_by_pelvis(kp3d_predict)
                
                kp3d_mpjpe = torch.norm(kp3d_predict - kp3d_real, dim=2).mean()
                
                aligned_kp3d = batch_compute_similarity_transform(kp3d_real, kp3d_predict)
                kp3d_mpjpe_aligned = torch.norm(aligned_kp3d - kp3d_real, dim=2).mean()
                
                self.kp3d_mpjpe_log.update(kp3d_mpjpe.item(), kp3d_real.shape[0])
                self.kp3d_mpjpe_aligned_log.update(kp3d_mpjpe_aligned.item(), kp3d_real.shape[0])

    def _log_val(self, epoch):
        self.summary_writer.add_scalar('val/kp2d_mpjpe', self.kp2d_mpjpe_log.result(), epoch)
        if self.config.USE_3D:
            self.summary_writer.add_scalar('val/kp3d_mpjpe', self.kp3d_mpjpe_log.result(), epoch)
            self.summary_writer.add_scalar('val/kp3d_mpjpe_aligned', self.kp3d_mpjpe_aligned_log.result(), epoch)
            
        self.kp2d_mpjpe_log.reset()
        if self.config.USE_3D:
            self.kp3d_mpjpe_log.reset()
            self.kp3d_mpjpe_aligned_log.reset()

    def test(self, return_kps=False):
        self.generator.eval()
        with torch.no_grad():
            dataset = Dataset()
            ds_test = dataset.get_test()
            mpjpe, mpjpe_aligned, sequences, kps3d_pred, kps3d_real = [], [], [], [], []
            
            for image_data in tqdm(ds_test, total=len(ds_test)):
                image, kp3d, sequence = image_data[0].to(self.device), image_data[1].to(self.device), image_data[2]
                kp3d_mpjpe_b, kp3d_mpjpe_aligned_b, predict_kp3d_b, real_kp3d_b = self._test_step(image, kp3d, return_kps=return_kps)
                
                if return_kps:
                    kps3d_pred.append(predict_kp3d_b.cpu())
                    kps3d_real.append(real_kp3d_b.cpu())
                mpjpe.append(kp3d_mpjpe_b.cpu())
                mpjpe_aligned.append(kp3d_mpjpe_aligned_b.cpu())
                sequences.append(sequence)

            def convert(tensor_list, num=None, is_kp=False):
                if not tensor_list: return torch.tensor([])
                tensor = torch.cat(tensor_list, dim=0)
                if is_kp:
                    return tensor.view(-1, num if num is not None else self.config.NUM_KP3D, 3)
                return tensor.view(-1, num if num is not None else 1)

            mpjpe, mpjpe_aligned, sequences = convert(mpjpe), convert(mpjpe_aligned), convert(sequences, 1)
            result_dict = {"kp3d_mpjpe": mpjpe, "kp3d_mpjpe_aligned": mpjpe_aligned, "seq": sequences}
            if return_kps:
                kps3d_pred, kps3d_real = convert(kps3d_pred, is_kp=True), convert(kps3d_real, is_kp=True)
                result_dict.update({'kps3d_pred': kps3d_pred, 'kps3d_real': kps3d_real})
            return result_dict

    def _test_step(self, image, kp3d, return_kps=False):
        if len(image.shape) != 4:
            image, kp3d = image.unsqueeze(0), kp3d.unsqueeze(0)
        
        result = self.generator(image)
        _, _, kp3d_pred, _, _, _ = result[-1]
        
        factor = 1000.0
        kp3d, kp3d_predict = kp3d * factor, kp3d_pred * factor
        kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]
        
        real_kp3d = batch_align_by_pelvis(kp3d)
        predict_kp3d = batch_align_by_pelvis(kp3d_predict)
        
        kp3d_mpjpe = torch.norm(real_kp3d - predict_kp3d, dim=2)
        aligned_kp3d = batch_compute_similarity_transform(real_kp3d, predict_kp3d)
        kp3d_mpjpe_aligned = torch.norm(real_kp3d - aligned_kp3d, dim=2)
        
        if return_kps:
            return kp3d_mpjpe, kp3d_mpjpe_aligned, predict_kp3d, real_kp3d
        return kp3d_mpjpe, kp3d_mpjpe_aligned, None, None

    def detect(self, image):
        self.generator.eval()
        with torch.no_grad():
            if len(image.shape) != 4:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            result = self.generator(image)
            vertices_pred, kp2d_pred, kp3d_pred, pose_pred, shape_pred, cam_pred = result[-1]
            result_dict = {
                "vertices": vertices_pred.squeeze(), "kp2d": kp2d_pred.squeeze(),
                "kp3d": kp3d_pred.squeeze(), "pose": pose_pred.squeeze(),
                "shape": shape_pred.squeeze(), "cam": cam_pred.squeeze()
            }
            return result_dict