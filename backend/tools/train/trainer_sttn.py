import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from backend.inpaint.sttn.auto_sttn import Discriminator
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.tools.train.dataset_sttn import Dataset
from backend.tools.train.loss_sttn import AdversarialLoss


class Trainer:
    def __init__(self, config, debug=False):
        # Trainer initialization
        self.config = config  # Save config info
        self.epoch = 0  # Current training epoch
        self.iteration = 0  # Current training iteration
        if debug:
            # If debug mode, set more frequent save and validation frequency
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # Set up dataset and dataloader
        self.train_dataset = Dataset(config['data_loader'], split='train', debug=debug)  # Create training set object
        self.train_sampler = None  # Initialize training set sampler as None
        self.train_args = config['trainer']  # Training process parameters
        if config['distributed']:
            # If distributed training, initialize distributed sampler
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank']
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),  # Shuffle if no sampler
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler
        )

        # Set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])  # Adversarial loss
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])  # Move loss function to corresponding device
        self.l1_loss = nn.L1Loss()  # L1 loss

        # Initialize generator and discriminator models
        self.netG = InpaintGenerator()  # Generator network
        self.netG = self.netG.to(self.config['device'])  # Move to device
        self.netD = Discriminator(
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge'
        )
        self.netD = self.netD.to(self.config['device'])  # Discriminator network
        # Initialize optimizers
        self.optimG = torch.optim.Adam(
            self.netG.parameters(),  # Generator parameters
            lr=config['trainer']['lr'],  # Learning rate
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2'])
        )
        self.optimD = torch.optim.Adam(
            self.netD.parameters(),  # Discriminator parameters
            lr=config['trainer']['lr'],  # Learning rate
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2'])
        )
        self.load()  # Load model

        if config['distributed']:
            # If distributed training, use distributed data parallel wrapper
            self.netG = DDP(
                self.netG,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True,
                find_unused_parameters=False
            )
            self.netD = DDP(
                self.netD,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True,
                find_unused_parameters=False
            )

        # Set logger
        self.dis_writer = None  # Discriminator writer
        self.gen_writer = None  # Generator writer
        self.summary = {}  # Store summary stats
        if self.config['global_rank'] == 0 or (not config['distributed']):
            # If not distributed training or is master node of distributed training
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis')
            )
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen')
            )

    # Get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

    # Adjust learning rate
    def adjust_learning_rate(self):
        # Calculate decayed learning rate
        decay = 0.1 ** (min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        # If new learning rate is different from current learning rate, update learning rate in optimizer
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # Add summary info
    def add_summary(self, writer, name, val):
        # Add and update statistical info, accumulate every iteration
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        # Record every 100 iterations
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    # Load model netG and netD
    def load(self):
        model_path = self.config['save_dir']  # Model save path
        # Check if latest model checkpoint exists
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            # Read last epoch number
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            # If latest.ckpt does not exist, try reading stored model file list to get the latest one
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()  # Sort model files to get the latest one
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None  # Get latest epoch value
        if latest_epoch is not None:
            # Construct generator and discriminator model file paths
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            # If master node, output loading model info
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            # Load generator model
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            # Load discriminator model
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            # Load optimizer state
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            # Update current epoch and iteration count
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            # If no model file found, output warning info
            if self.config['global_rank'] == 0:
                print('Warning: There is no trained model found. An initialized model will be used.')

    # Save model parameters, called once every evaluation cycle (eval_epoch)
    def save(self, it):
        # Execute save operation only on process with global rank 0, usually representing master node
        if self.config['global_rank'] == 0:
            # Generate file path to save generator model state dictionary
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            # Generate file path to save discriminator model state dictionary
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            # Generate file path to save optimizer state dictionary
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))

            # Print message indicating model is being saved
            print('\nsaving model to {} ...'.format(gen_path))

            # Determine if model is wrapped by DataParallel or DDP, if so get original model
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD

            # Save generator and discriminator model parameters
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            # Save current epoch, iteration count and optimizer state
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict()
            }, opt_path)

            # Write latest iteration count to "latest.ckpt" file
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

        # Training entry
    def train(self):
        # Initialize progress bar range
        pbar = range(int(self.train_args['iterations']))
        # If global rank 0 process, set to display progress bar
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        # Start training loop
        while True:
            self.epoch += 1  # epoch count increase
            if self.config['distributed']:
                # If distributed training, set sampler to ensure each process gets different data
                self.train_sampler.set_epoch(self.epoch)

            # Call function to train one epoch
            self._train_epoch(pbar)
            # If iteration count exceeds iteration limit in config, exit loop
            if self.iteration > self.train_args['iterations']:
                break
        # Training end output
        print('\nEnd training....')

        # Process input and calculate loss for each training cycle
    def _train_epoch(self, pbar):
        device = self.config['device']  # Get device info

        # Traverse data in dataloader
        for frames, masks in self.train_loader:
            # Adjust learning rate
            self.adjust_learning_rate()
            # Iteration count +1
            self.iteration += 1

            # Move frames and masks to device
            frames, masks = frames.to(device), masks.to(device)
            b, t, c, h, w = frames.size()  # Get frame and mask sizes
            masked_frame = (frames * (1 - masks).float())  # Apply mask to image
            pred_img = self.netG(masked_frame, masks)  # Use generator to generate filled image
            # Adjust frames and masks dimensions to match network input requirements
            frames = frames.view(b * t, c, h, w)
            masks = masks.view(b * t, 1, h, w)
            comp_img = frames * (1. - masks) + masks * pred_img  # Generate final combined image

            gen_loss = 0  # Initialize generator loss
            dis_loss = 0  # Initialize discriminator loss

            # Discriminator adversarial loss
            real_vid_feat = self.netD(frames)  # Discriminator discriminates real image
            fake_vid_feat = self.netD(comp_img.detach())  # Discriminator discriminates generated image, note detach to not calculate gradients
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)  # Real image loss
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)  # Generated image loss
            dis_loss += (dis_real_loss + dis_fake_loss) / 2  # Average discriminator loss
            # Add discriminator loss to summary
            self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
            self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            # Optimize discriminator
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # Generator adversarial loss
            gen_vid_feat = self.netD(comp_img)
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)  # Generator adversarial loss
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']  # Weight amplification
            gen_loss += gan_loss  # Accumulate to generator loss
            # Add generator adversarial loss to summary
            self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # Generator L1 loss
            hole_loss = self.l1_loss(pred_img * masks, frames * masks)  # Calculate loss only for masked area
            # Consider mean of mask, multiply by hole_weight in config
            hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            gen_loss += hole_loss  # Accumulate to generator loss
            # Add hole_loss to summary
            self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())

            # Calculate L1 loss for area outside mask
            valid_loss = self.l1_loss(pred_img * (1 - masks), frames * (1 - masks))
            # Consider mean of non-masked area, multiply by valid_weight in config
            valid_loss = valid_loss / torch.mean(1 - masks) * self.config['losses']['valid_weight']
            gen_loss += valid_loss  # Accumulate to generator loss
            # Add valid_loss to summary
            self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())

            # Generator optimization
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # Console log output
            if self.config['global_rank'] == 0:
                pbar.update(1)  # Progress bar update
                pbar.set_description((  # Set progress bar description
                    f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"  # Print loss values
                    f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}")
                )

            # Model save
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration // self.train_args['save_freq']))
            # Iteration count termination judgment
            if self.iteration > self.train_args['iterations']:
                break

