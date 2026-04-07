import torch 
import torch.nn as nn 
from datetime import datetime
from pathlib import Path 
from typing import Dict
from src.misc import dist
from src.core import BaseConfig
class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg 
    def setup(self, ):
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch
        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)
        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    def train(self, ):
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)
        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader,            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader,            shuffle=self.cfg.val_dataloader.shuffle)
    def eval(self, ):
        self.setup()
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader,            shuffle=self.cfg.val_dataloader.shuffle)
        if self.cfg.infer:
            self.infer_dataloader = dist.warp_loader(self.cfg.infer_dataloader)
        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)
    def state_dict(self, last_epoch):
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()
        state['last_epoch'] = last_epoch
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    def load_state_dict(self, state):
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')
        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                model_state_dict = self.model.state_dict()
                adas1=[k for k in model_state_dict.keys() if 'Adapter' in k ]
                adas2=[k for k in state['model'].keys() if 'Adapter' in k ]
                print(len(adas1),'adpter keys number vs',len(adas2))
                model_keys = set(adas1)
                loaded_keys = set(adas2)
                missing_keys = model_keys - loaded_keys
                unexpected_keys = loaded_keys - model_keys
                if missing_keys:
                    print('以下参数在加载的权重中缺失：')
                    for key in missing_keys:
                        print(key)
                else:
                    print('加载的权重中没有缺失的参数。')
                if unexpected_keys:
                    print('以下参数在模型中不存在，但在加载的权重中存在：')
                    for key in unexpected_keys:
                        print(key)
                else:
                    print('加载的权重中没有多余的参数。')
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')
        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')
        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')
        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')
        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')
    def save(self, path):
        state = self.state_dict()
        dist.save_on_master(state, path)
    def resume(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)
    def load_tuning_state(self, path,):
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')
        module = dist.de_parallel(self.model)
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])
        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')
    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)
        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}
    def fit(self, ):
        raise NotImplementedError('')
    def val(self, ):
        raise NotImplementedError('')