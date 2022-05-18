import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.utils import setup_logger

from transformers import AdamW, AutoTokenizer, AutoConfig

import logging
from pprint import pformat
import argparse
import os
import json
import math
from pathlib import Path
from functools import partial

from utils.switch import get_modules, get_train_aux
from utils.auxiliary import set_seed, average_distributed_scalar
from utils.argument import verify_args, update_additional_params, set_default_params, set_default_dataset_params


logger = logging.getLogger(__file__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--params_file", type=str, help="JSON configuration file")
  # parser.add_argument("--dataset_path", type=str, default="data",
  #                     help="Path of the dataset.")
  # parser.add_argument("--dataset_cache", type=str, default='data/dataset_cache', help="Path of the dataset cache")
  parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the model")
  parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
  parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
  parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                      help="Accumulate gradients on several steps")
  parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
  parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
  parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
  parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
  parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
  parser.add_argument("--eval_before_start", action='store_true',
                      help="If true start with a first evaluation before training")
  parser.add_argument("--save_every_iters", type=int, default=None, help="Number of training iters to save")
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training (-1: not distributed)")
  parser.add_argument("--fp16", type=str, default="",
                      help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
  parser.add_argument("--resume_from", type=str, default=None, help="resume training.")
  parser.add_argument("--seed", type=int, default=43)
  parser.add_argument("--debug", action='store_true')
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

  # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
  # logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
  logger.info("Arguments: %s", pformat(args))

  args.output_path = os.path.join('runs', args.output_path)
  
  if args.resume_from is not None and os.path.split(args.resume_from)[0] == '':
    args.resume_from = os.path.join(args.output_path, args.resume_from)

  # Setup CUDA, GPU & distributed training
  args.distributed = (args.local_rank != -1)
  if not args.distributed:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
  args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
  args.device = device

  # Set seed
  set_seed(args)
  # logger = setup_logger("trainer", distributed_rank=args.local_rank)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

  # Tokenizer construction
  config = AutoConfig.from_pretrained(args.model_name_or_path)
  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  try:
    SPECIAL_TOKENS = dataloader.SPECIAL_TOKENS
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
  except:
    pass
  args._tokenizer = tokenizer

  # Dataset construction
  dataset_class = getattr(dataloader, args.dataloader_class)
  if args.debug:
    train_dataset = dataset_class(args, tokenizer, 'valid')
  else:
    train_dataset = dataset_class(args, tokenizer, 'train')
  valid_dataset = dataset_class(args, tokenizer, 'valid')
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
  valid_sampler = RandomSampler(valid_dataset) if args.local_rank == -1 else DistributedSampler(valid_dataset)
  valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, collate_fn=valid_dataset.collate_fn)
  steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
  args.steps_per_epoch = steps_per_epoch

  # Model construction
  model_class = getattr(models, args.model_class)
  if hasattr(args, "decoder_model_name_or_path"):
    from transformers.modeling_encoder_decoder import EncoderDecoderConfig
    config_encoder = config_decoder = config
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = model_class.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=args.model_name_or_path, 
            decoder_pretrained_model_name_or_path=args.decoder_model_name_or_path, 
            args=args
          )
  else:
    if 'gpt' in args.model_class.lower():
      model = model_class.from_pretrained(args.model_name_or_path, config=config)
    else:
      model = model_class.from_pretrained(args.model_name_or_path, config=config, args=args)
  model.resize_token_embeddings(len(tokenizer))
  model = model.to(device)

  logger.info(f"Count parameters: {count_parameters(model)}")

  if args.local_rank == 0:
    torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

  # Optimizer, lr_scheduler
  optimizer, lr_scheduler = get_train_aux(args, model)
  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  # Training function and trainer
  update = partial(helper.trainer_update, args=args, model=model, optimizer=optimizer)
  trainer = Engine(update)

  to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
  train_metrics = helper.train_metrics

  common.setup_common_training_handlers(
    trainer=trainer,
    train_sampler=train_loader.sampler,
    to_save=to_save,
    save_every_iters=args.save_every_iters or len(train_loader),
    lr_scheduler=None,
    output_names=train_metrics,
    with_pbars=False,
    clear_cuda_cache=False,
    output_path=args.output_path,
    n_saved=1,
  )
  trainer.add_event_handler(Events.ITERATION_STARTED(every=args.gradient_accumulation_steps), lr_scheduler)

  resume_from = args.resume_from
  if resume_from is not None:
    checkpoint_fp = Path(resume_from)
    assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
    logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    to_load = to_save
    if 'validation' in resume_from:
      to_load = {"model": model}
    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

  # Prepare metrics - note how we compute distributed metrics
  _inference = partial(helper.evaluator_update, args=args, model=model)
  evaluator = Engine(_inference)
  metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0][0], x[1][0]))}
  metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
  metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
  metrics["neg_ppl"] = MetricsLambda(lambda x: -x, metrics["average_ppl"])
  for name, metric in metrics.items():
    metric.attach(evaluator, name)

  trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
  if args.n_epochs < 1:
    trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(valid_loader))
  if args.eval_before_start:
    trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(valid_loader))

  # Make sure distributed data samplers split the dataset nicely between the distributed processes
  if args.distributed:
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

  # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
  if args.local_rank in [-1, 0]:
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=train_metrics, output_transform=lambda _: {"lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

    tb_logger = common.setup_tb_logging(args.output_path, trainer, optimizer, evaluators={'validation': evaluator}, log_every_iters=1)

    # Good practice: save your training arguments together with the trained model
    @trainer.on(Events.COMPLETED)
    def save_args():
      torch.save(args, os.path.join(args.output_path, "training_args.bin"))
      with open(os.path.join(args.output_path, "params.json"), "w") as jsonfile:
          json.dump(params, jsonfile, indent=2)


  # Store 3 best models by validation accuracy:
  common.gen_save_best_models_by_val_score(
    save_handler=DiskSaver(args.output_path, require_empty=False),
    evaluator=evaluator,
    models={"model": model},
    metric_name="neg_ppl",
    n_saved=1,
    trainer=trainer,
    tag="validation"
  )

  # Run the training
  trainer.run(train_loader, max_epochs=args.n_epochs)

  if args.local_rank in [-1, 0]:
    tb_logger.close()

if __name__ == '__main__':
  main()