import time 
import json
import datetime
import torch 
from src.misc import dist
from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate,  evaluate_x
def _get_coco_api_from_dataset(dataset):
    from src.data import get_coco_api_from_dataset
    return get_coco_api_from_dataset(dataset)
class DetSolver(BaseSolver):
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg 
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        base_ds = _get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, }
        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)
            self.lr_scheduler.step()
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'best_checkpoint.pth']
                checkpoint_step = getattr(args, 'checkpoint_step', 3)  
                if (epoch + 1) % checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)
            if epoch >= 20:  
                module = self.ema.module if self.ema else self.model
                try:
                    test_stats, coco_evaluator = evaluate(
                        module, self.criterion, self.postprocessor, self.val_dataloader, 
                        base_ds, self.device, self.output_dir,
                        annot_file=self.cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'],
                        outcsv_filename=self.cfg.outcsv_filename,
                    )
                except Exception as e:
                    print(f"Warning: Validation failed at epoch {epoch}: {e}")
                    print("Training will continue despite validation error")
                    test_stats = {}
                    coco_evaluator = None
                for k in test_stats.keys():
                    if test_stats[k] !=[]:
                        if k in best_stat:
                            best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                            best_stat[k] = max(best_stat[k], test_stats[k][0])
                        else:
                            best_stat['epoch'] = epoch
                            print(list(test_stats.keys()))
                            print(k,"xxxxxx",test_stats[k])
                            best_stat[k] = test_stats[k][0]
                print('best_stat: ', best_stat)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
                if self.output_dir and dist.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                        self.output_dir / "eval" / name)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    def val(self, ):
        self.eval()
        base_ds = _get_coco_api_from_dataset(self.val_dataloader.dataset)
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir, 
                annot_file=self.cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'],
                outcsv_filename=self.cfg.outcsv_filename,
                )
        debug=True
        if debug:
            epoch=999
            val_best_stat = {'epoch': -1, }
            for i, k in enumerate(test_stats.keys()):
                print(test_stats,f"testing::@val {i}::{test_stats}")
                if test_stats[k] !=[]:
                    if k in val_best_stat:
                        val_best_stat['epoch'] = epoch if test_stats[k][0] > val_best_stat[k] else val_best_stat['epoch']
                        val_best_stat[k] = max(val_best_stat[k], test_stats[k][0])
                    else:
                        val_best_stat['epoch'] = epoch
                        print(k,"xxxxxx",test_stats[k])
                        val_best_stat[k] = test_stats[k][0]
            print('val_best_stat@val(): ', val_best_stat)
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return
    def infer(self, ):
        self.eval()
        assert self.cfg.infer_dataloader, f'built the infer_dataloader with csv file and image directory firstly!!!'
        module = self.ema.module if self.ema else self.model
        evaluate_x(module, self.criterion, self.postprocessor,
                self.infer_dataloader, self.device,
                outcsv_filename=self.cfg.outcsv_filename,
                visual_check=self.cfg.visual_check,
                other2ppsocr=self.cfg.other2ppsocr,
                getacc=self.cfg.getacc,
                )
        return 