import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import torch
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
os.environ["CUDA_VISIBLE_DEVICES"]='3'
torch.multiprocessing.set_sharing_strategy("file_system")
def main(args, ) -> None:
    dist.init_distributed()
    assert not all([args.tuning, args.resume]),        'Only support from_scrach or resume or tuning at one time'
    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    cfg.outcsv_filename=args.outcsv_filename
    cfg.output_dir=args.output_dir
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/moldetr/moldetr_r50vd_6x_coco.yml')
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--tuning', '-t', type=str, default=None)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--outcsv_filename', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    args.output_dir     ="output20260407_to_del"
    if args.outcsv_filename is None:
        args.outcsv_filename = 'output0602/R4.csv'
    if args.output_dir is None:
        args.output_dir = "output0602"
    main(args)
