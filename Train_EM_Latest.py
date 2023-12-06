# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.utils import setup_logger
import random

from transformers import AdamW, AutoConfig, BartTokenizer, BartTokenizerFast
# BartTokenizerFast,AutoTokenizer
import logging
from pprint import pformat
import argparse
import os
import json
import copy
import math
from pathlib import Path
from functools import partial
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from utils.switch import get_modules, get_train_aux
from utils.auxiliary import set_seed, average_distributed_scalar
from utils.argument import verify_args, update_additional_params, set_default_params, set_default_dataset_params

logger = logging.getLogger(__file__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")
    parser.add_argument("--dataset_path", type=str,
                        default="", help="Path of the dataset.")
    parser.add_argument("--model_checkpoint", type=str,
                        default="", help="Path, url or short name of the model")
    parser.add_argument("--output_path", type=str,
                        required=True, help="Path to save the model")
    parser.add_argument("--max_history", type=int, default=2,
                        help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float,
                        default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float,
                        default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0,
                        help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float,
                        default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--MAX_UTTERANCE_NUM", type=int,
                        default=7, help="MAX_UTTERANCE_NUM")
    parser.add_argument("--MAX_SPEAKER_NUM", type=int,
                        default=8, help="MAX_SPEAKER_NUM")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--save_every_iters", type=int,
                        help="Number of training iters to save")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--resume_from", type=str,
                        default=None, help="resume training.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--checkpoint_output_path", type=str,
                        default="", help="Path to save the checkpoint")
    parser.add_argument("--em_batch_size", type=int,
                        default=4, help="Batch size for EM")
    parser.add_argument("--iterations", type=int, default=1,
                        help="outer shuffle iterations for training")
    parser.add_argument("--em_iterations", type=int,
                        default=6, help="iterations for em_training")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        # update_additional_params(params, args)
        args.update(params)
        args = argparse.Namespace(**args)

    dataloader, models, helper = get_modules(args)

    args.output_path = os.path.join('runs', args.output_path)

    logger.warning("Running process %d",
                   args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        # print("device is ok ...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device
    # print("device=",args.device)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    # Tokenizer construction
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BartTokenizerFast.from_pretrained(args.model_name_or_path)
    print("config and tokenizer is ok ...")

    try:
        SPECIAL_TOKENS = dataloader.SPECIAL_TOKENS
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
    except:
        pass
    args._tokenizer = tokenizer

    model_class = getattr(models, args.model_class)
    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, args=args)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)

    checkpoint_fp = Path(args.model_checkpoint)
    if checkpoint_fp.is_dir():
        checkpoint_fp = max(filter(lambda x: x.name.startswith("training_checkpoint_"),
                                   checkpoint_fp.iterdir()), key=lambda x: int(x.stem.split('_')[-1]))
    assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(
        checkpoint_fp.as_posix())
    logger.info("Resume from a checkpoint: {}".format(
        checkpoint_fp.as_posix()))
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    print("checkpoint is ok ...")

    dataset_class = getattr(dataloader, args.dataloader_class)
    # print("dataset_class=",dataset_class)
    valid_dataset = dataset_class(args, tokenizer, 'valid', line_batch_list=[])
    valid_sampler = RandomSampler(
        valid_dataset) if args.local_rank == -1 else DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size,
                              collate_fn=valid_dataset.collate_fn)
    print("valid_loader is ok ...")

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        print("now is DistributedDataParallel ...")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)

    # lines = [json.loads(line) for line in open(args.dataset_path, 'r', encoding='utf-8')]
    # print("len(lines)=",len(lines))

    # # # Training function and trainer
    update = partial(helper.trainer_update, args=args,
                     model=model, optimizer=optimizer)
    trainer = Engine(update)

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer}
    train_metrics = helper.train_metrics
    print("trainer and is ok...")

    # metric
    # # Prepare metrics - note how we compute distributed metrics  #### ppl
    _inference = partial(helper.evaluator_update, args=args, model=model)
    evaluator = Engine(_inference)
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (
        x[0][0], x[1][0]))}  # (lm_logits_flat_shifted, ), (lm_labels_flat_shifted, )
    metrics.update({"average_nll": MetricsLambda(
        average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    metrics["neg_ppl"] = MetricsLambda(lambda x: -x, metrics["average_ppl"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    print("evaluator metrics is ok ...")

    # Store 3 best models by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=DiskSaver(args.output_path, require_empty=False),
        evaluator=evaluator,
        models={"model": model},
        metric_name="neg_ppl",
        n_saved=1,
        trainer=trainer,
        tag="validation")
    print("common is ok ...")

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(valid_loader))

    if args.n_epochs < 1:
        trainer.add_event_handler(
            Events.COMPLETED, lambda _: evaluator.run(valid_loader))
    if args.eval_before_start:
        trainer.add_event_handler(
            Events.STARTED, lambda _: evaluator.run(valid_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED,
                                  lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED,
                                    lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=train_metrics,
                    output_transform=lambda _: {"lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(
            "Validation: %s" % pformat(evaluator.state.metrics)))

        @trainer.on(Events.COMPLETED)
        def save_args():
            torch.save(args, os.path.join(
                args.output_path, "training_args.bin"))
            with open(os.path.join(args.output_path, "params.json"), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)
    print("event is ok ...")

    for iter in range(args.iterations):
        random.shuffle(lines)
        t = 0
        # E步
        for line in range(0, len(lines), args.em_batch_size):
            batch_list_train = lines[line:line + args.em_batch_size]
            t += 1

            for em_iter in range(args.em_iterations):
                train_batch_list = []
                print("now is %s em_iters" % em_iter)

                for example in tqdm(batch_list_train):
                    idx = 0
                    if -1 not in example["ctx_adr"]:
                        train_batch_list.append(example)
                    elif example["ctx_adr"][0] == -1 and -1 not in example["ctx_adr"][1:]:
                        train_batch_list.append(example)
                    else:
                        for i in range(len(example["ctx_adr"]) - 1, 0, -1):
                            idx += 1
                            if example["ctx_adr"][i] == -1:
                                idx_noadr_near = len(example["ctx_adr"]) - idx
                                batch_list = []
                                for i, m in enumerate(example["ctx_spk"][0:idx_noadr_near]):
                                    new_example = copy.deepcopy(example)
                                    new_example["ctx_adr"][idx_noadr_near] = m
                                    new_example["relation_at"].append(
                                        [idx_noadr_near, i])
                                    batch_list.append(new_example)

                                example_dataset = dataset_class(
                                    args, tokenizer, "train", batch_list)
                                example_dataloader = DataLoader(example_dataset,
                                                                batch_size=len(batch_list), collate_fn=example_dataset.collate_fn)

                                for batch_example in example_dataloader:
                                    model.eval()
                                    with torch.no_grad():
                                        batch = tuple(input_tensor.to(
                                            args.device) for input_tensor in batch_example)
                                        graph, input_ids, decode_input, segment_id, input_masks, lm_labels, ans_idxs, ans_from = batch
                                        outputs = model(graph=graph, input_ids=input_ids, decoder_input_ids=decode_input,
                                                        token_type_ids=None, labels=lm_labels, attention_mask=input_masks,
                                                        decoder_ans_idxs=ans_idxs, decoder_ans_from=ans_from, return_dict=True)
                                        # [bz,len,vocab_size]
                                        lm_logits = outputs.logits
                                        word_logits = lm_logits.gather(
                                            dim=2, index=lm_labels.unsqueeze(-1)).squeeze()  # (bsz, len)
                                        summ_logprobs = word_logits.sum(
                                            dim=-1)  # (bsz)
                                        P_max_idx = torch.argmax(
                                            summ_logprobs, dim=0)
                                        P_max_idx = P_max_idx.cpu().numpy()
                                        train_batch_list.append(
                                            batch_list[P_max_idx])

                                break

                # M步
                print("len(train_batch_list)=", len(train_batch_list))
                if len(train_batch_list) == args.em_batch_size:
                    train_dataset = dataset_class(
                        args, tokenizer, "train", train_batch_list)
                    train_sampler = RandomSampler(
                        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
                    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                              collate_fn=train_dataset.collate_fn)

                    if args.distributed:
                        trainer.add_event_handler(Events.EPOCH_STARTED,
                                                  lambda engine: train_sampler.set_epoch(engine.state.epoch))
                        evaluator.add_event_handler(Events.EPOCH_STARTED,
                                                    lambda engine: valid_sampler.set_epoch(engine.state.epoch))

                    common.setup_common_training_handlers(
                        trainer=trainer,
                        train_sampler=train_loader.sampler,
                        to_save=to_save,
                        save_every_iters=args.save_every_iters or len(
                            train_loader),
                        lr_scheduler=None,
                        output_names=train_metrics,
                        with_pbars=False,
                        clear_cuda_cache=False,
                        output_path=args.output_path,
                        n_saved=1,
                    )

                    resume_from = args.resume_from
                    if resume_from is not None:
                        checkpoint_fp = Path(resume_from)

                        if t > 1 or em_iter > 0:
                            if checkpoint_fp.is_dir():
                                checkpoint_fp = max(filter(lambda x: x.name.startswith("training_checkpoint_"),
                                                           checkpoint_fp.iterdir()), key=lambda x: int(x.stem.split('_')[-1]))
                                logger.info("Resume from a checkpoint: {}".format(
                                    checkpoint_fp.as_posix()))
                                checkpoint = torch.load(
                                    checkpoint_fp.as_posix(), map_location="cpu")
                                to_load = to_save
                                if 'validation' in resume_from:
                                    to_load = {"model": model}
                                Checkpoint.load_objects(
                                    to_load=to_load, checkpoint=checkpoint)
                                print("Checkpoint load_objects is ok ...")

                    trainer.run(train_loader, max_epochs=args.n_epochs)

                else:
                    raise ValueError(
                        "len(train_batch_list) is != em_batch_size!")

    print("model training is finished ...")


if __name__ == '__main__':
    main()
