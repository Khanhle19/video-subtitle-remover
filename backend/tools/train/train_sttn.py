import os
import json
import argparse
from shutil import copyfile
import torch
import torch.multiprocessing as mp

from backend.tools.train.trainer_sttn import Trainer
from backend.tools.train.utils_sttn import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs_sttn/youtube-vos.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
args = parser.parse_args()


def main_worker(rank, config):
    # If local ranking (local_rank) is not mentioned in config, assign it and global ranking (global_rank) to passed ranking (rank)
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank

    # If config specifies distributed training
    if config['distributed']:
        # Set CUDA device to GPU corresponding to current local rank
        torch.cuda.set_device(int(config['local_rank']))
        # Initialize distributed process group via nccl backend
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=config['init_method'],
            world_size=config['world_size'],
            rank=config['global_rank'],
            group_name='mtorch'
        )
        # Print current GPU usage, output global rank and local rank
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank']))
        )

    # Create model save directory path, including model name and config file name
    config['save_dir'] = os.path.join(
        config['save_dir'], '{}_{}'.format(config['model'], os.path.basename(args.config).split('.')[0])
    )

    # If CUDA is available, set device to corresponding CUDA device, otherwise CPU
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    # If not distributed training, or is master node of distributed training (rank 0)
    if (not config['distributed']) or config['global_rank'] == 0:
        # Create model save directory, allow ignoring creation if directory exists (exist_ok=True)
        os.makedirs(config['save_dir'], exist_ok=True)
        # Set config file save path
        config_path = os.path.join(
            config['save_dir'], config['config'].split('/')[-1]
        )
        # If config file does not exist, copy from given config path to new path
        if not os.path.isfile(config_path):
            copyfile(config['config'], config_path)
        # Print directory creation info
        print('[**] create folder {}'.format(config['save_dir']))

    # Initialize trainer, pass config parameters and debug flag
    trainer = Trainer(config, debug=args.exam)
    # Start training
    trainer.train()


if __name__ == "__main__":
    # Load config file
    config = json.load(open(args.config))
    config['model'] = args.model  # Set model name
    config['config'] = args.config  # Set config file path

    # Set distributed training related config
    config['world_size'] = get_world_size()  # Get global process count, i.e., total GPU count participating in training
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"  # Set initialization method, including master node IP and port
    config['distributed'] = True if config['world_size'] > 1 else False  # Determine whether to enable distributed training based on world size

    # Set distributed parallel training environment
    if get_master_ip() == "127.0.0.1":
        # If master node IP is local address, manually spawn multiple distributed training processes
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
    else:
        # If started by other tools like OpenMPI, no need to manually create processes.
        config['local_rank'] = get_local_rank()  # Get local (single node) rank
        config['global_rank'] = get_global_rank()  # Get global rank
        main_worker(-1, config)  # Start main worker function
