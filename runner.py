# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import wandb
from dist_utils import get_rank, init_distributed_mode
from dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from logger import MetricLogger, SmoothedValue
from utils import get_dataloader, prepare_sample
from optims import get_optimizer, LinearWarmupCosineLRScheduler
# from metrics import compute_wer, compute_spider
from custom_metrics import compute_per, compute_f1, compute_wer, compute_spider, compute_meteor


class Runner:
    def __init__(self, cfg, model, datasets, job_id, dryrun, stage = "stage1"):
        self.config = cfg

        # dryrun (test with dummy model)
        self.dryrun = dryrun

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.device = torch.device(self.config.config.run.device)
        self.use_distributed = self.config.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.evaluate_only = self.config.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")
        self.stage = stage
        self.best_path = ""
        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r") as f:
                    self.test_prompt_dict = json.load(f)
            except:
                print("Failed to load test prompt! Try to use utf-8 encoding.")
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            for k in self.test_prompt_dict.keys():
                self.test_prompt_dict[k] = self.prompt_template.format(self.test_prompt_dict[k])

        else:
            self.test_prompt_dict = None

        # model
        self._model = model
        self._model.to(self.device)

        if self.use_distributed:
            self.model = DDP(
                self._model, device_ids=[self.config.config.run.gpu],
                find_unused_parameters=True
            )
        else:
            self.model = self._model
        # dataloaders
        self.train_loader = get_dataloader(datasets["train"], self.config.config.run, is_train=True, use_distributed=self.use_distributed)
        self.valid_loader = get_dataloader(datasets["valid"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)
        self.test_loader = get_dataloader(datasets["test"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)

        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # optimizer & scheduler
        self.iters_per_epoch = len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        self.log_config()
        
    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        for i in metric_logger.log_every(range(self.iters_per_epoch), self.config.config.run.log_freq, header=header, logger=self.log_writter, start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break
            
            samples = next(self.train_loader)
            # print("Memory before sample", torch.cuda.memory_allocated(dist.get_rank()) / 1024**2, "MB")
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
            # print("Memory after sample", torch.cuda.memory_allocated(dist.get_rank()) / 1024**2, "MB")
            
            if not self.dryrun:
                self.scheduler.step(cur_epoch=epoch, cur_step=i)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.model(samples)["loss"]

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                
                global_rank = int(os.environ.get("RANK", 0))
                if global_rank == 0:
                    wandb.log({"train/iteration": i, "train/loss": loss.item(), "train/lr": self.optimizer.param_groups[0]["lr"]})
            else: # dryrun, no model availble
                metric_logger.update(loss=0.0)
                metric_logger.update(lr=0.0)
                global_rank = int(os.environ.get("RANK", 0))
                if global_rank == 0:
                    wandb.log({"train/iteration": i, "train/loss": 0.0, "train/lr": 0.0})

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=False, save_json=False):
        if not self.dryrun:
            model = self.unwrap_dist_model(self.model)
            model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)

        results = []
        for samples in metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            if not self.dryrun:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    forward_result = model(samples, verbose=True)
                loss = forward_result.get("loss", 0)
                correct = forward_result.get("correct", 0)
                total = forward_result.get("total", 1)
                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": loss.item(),
                    "acc": (correct / total).item(),
                    "total": total,
                    "task": samples["task"],  # task 정보 추가
                }
            else:
                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": 0.0,
                    "acc": 0.0,
                    "total": 1,
                    "task": samples["task"],  # task 정보 추가
                }

            if decode:
                if model.prompt_dict:
                    if self.test_prompt_dict is None:
                        prompts = None
                    else:
                        prompts = [self.test_prompt_dict[s] for s in samples["task"]]
                        if "Q" in samples:
                            prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
                else:
                    prompts = None

                text = model.generate(samples, self.config.config.run, prompts=prompts)
                res["text"] = text
                res["prompt"] = prompts

            results.append(res)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            self.save_result(results, self.output_dir, "eval_{}_epoch_{}".format(split, epoch))

        # 기존 메트릭 계산 (loss, agg_metrics)
        res = {
            "loss": torch.tensor(0).float().cuda(),
            "n_sample": torch.tensor(0).float().cuda(),
            "correct": torch.tensor(0).float().cuda(),
            "n_token": torch.tensor(0).float().cuda(),
        }

        # Task별 메트릭 계산을 위한 초기화
        task_metrics = {
            "asr": {"wer": 0.0, "n_sample": 0},
            "audiocaption": {"meteor": 0.0, "n_sample": 0},
            "QA": {"f1": 0.0, "n_sample": 0},
            "phone_recognition": {"per": 0.0, "n_sample": 0},
            "audiocaption_v2": {"meteor": 0.0, "n_sample": 0},
            "gender_recognition": {"f1": 0.0, "n_sample": 0},
        }

        # 각 샘플에 대해 기존 메트릭 및 task별 메트릭 계산
        for item in results:
            # 기존 메트릭 계산
            item_loss = item["loss"]
            item_n_sample = len(item["id"])
            item_correct = item["acc"] * item["total"]
            item_n_token = item["total"]
            res["loss"] += item_loss * item_n_sample
            res["n_sample"] += item_n_sample
            res["correct"] += item_correct
            res["n_token"] += item_n_token

            # Task별 메트릭 계산
            tasks = item["task"]

            # tasks, text, ground_truth가 모두 같은 순서로 대응한다고 가정합니다.
            for i, task in enumerate(tasks):
                if task in task_metrics:
                    if task == "asr":
                        wer = compute_wer(item["text"][i], item["ground_truth"][i])
                        if wer is None:
                            wer = 0.0
                        task_metrics[task]["wer"] += wer  # 각 문장별 계산 후 누적
                        task_metrics[task]["n_sample"] += 1
                    elif task in ["audiocaption", "audiocaption_v2"]:
                        meteor_score = compute_meteor(item["text"][i], item["ground_truth"][i])
                        if meteor_score is None:
                            meteor_score = 0.0
                        task_metrics[task]["meteor"] += meteor_score
                        task_metrics[task]["n_sample"] += 1
                    elif task in ["QA", "gender_recognition"]:
                        f1 = compute_f1(item["text"][i], item["ground_truth"][i])
                        if f1 is None:
                            f1 = 0.0
                        task_metrics[task]["f1"] += f1
                        task_metrics[task]["n_sample"] += 1
                    elif task == "phone_recognition":
                        per = compute_per(item["text"][i], item["ground_truth"][i])
                        if per is None:
                            per = 0.0
                        task_metrics[task]["per"] += per
                        task_metrics[task]["n_sample"] += 1


        # 분산 환경에서 모든 프로세스의 결과 합산
        if is_dist_avail_and_initialized():
            dist.all_reduce(res["loss"])
            dist.all_reduce(res["n_sample"])
            dist.all_reduce(res["correct"])
            dist.all_reduce(res["n_token"])

            for task, metrics in task_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        dist.all_reduce(value)

        # 기존 메트릭 계산
        ret = {"loss": 0, "agg_metrics": 0}
        ret["loss"] = (res["loss"] / res["n_sample"]).item()
        ret["agg_metrics"] = (res["correct"] / res["n_token"]).item()

        # Task별 평균 메트릭 계산
        for task, metrics in task_metrics.items():
            if metrics["n_sample"] > 0:
                if task == "asr":
                    ret[f"{task}_wer"] = metrics["wer"] / metrics["n_sample"]
                elif task == "audiocaption" or task == "audiocaption_v2":
                    ret[f"{task}_meteor"] = metrics["meteor"] / metrics["n_sample"]
                elif task == "QA" or task == "gender_recognition":
                    ret[f"{task}_f1"] = metrics["f1"] / metrics["n_sample"]
                elif task == "phone_recognition":
                    ret[f"{task}_per"] = metrics["per"] / metrics["n_sample"]

        return ret

    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=True, save_json=False)
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch

                        self.save_checkpoint(cur_epoch, is_best=True)                    

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")
                    # WandB에 로깅
                    wandb_log = {
                        "valid/epoch": cur_epoch,
                        "valid/agg_metrics": agg_metrics,
                        "valid/loss": valid_log["loss"],
                    }

                    # Task별 메트릭 추가
                    task_metrics = {
                        "asr": "wer",
                        "audiocaption": "meteor",
                        "audiocaption_v2": "meteor",
                        "QA": "f1",
                        "phone_recognition": "per",
                        "gender_recognition": "f1",
                    }

                    for task, metric_name in task_metrics.items():
                        if f"{task}_{metric_name}" in valid_log:
                            wandb_log[f"valid/{task}_{metric_name}"] = valid_log[f"{task}_{metric_name}"]

                    wandb.log(wandb_log)

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        # testing phase
        if self.evaluate_only:
            test_log = self.valid_epoch("best", "test", decode=True, save_json=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        if is_best :
            self.best_path = save_to
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
        
    def resume_from_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # 모델 상태 복원
        model_no_ddp = self.unwrap_dist_model(self.model)
        model_no_ddp.load_state_dict(checkpoint["model"])
        
        # Optimizer 상태 복원
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Scheduler 상태 복원 (존재할 경우)
        if "scheduler" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Scaler (혼합 정밀도 사용 시)
        if self.scaler is not None and checkpoint.get("scaler", None) is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        # 중단했던 epoch 값 복원 (checkpoint에 저장된 epoch 다음부터 재개)
        self.start_epoch = checkpoint["epoch"] + 1
        
        logging.info("Checkpoint {} loaded. Resuming training from epoch {}.".format(checkpoint_file, self.start_epoch))

