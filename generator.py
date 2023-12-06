import torch
from torch.utils.data import DataLoader
from ignite.handlers import Checkpoint

from transformers import AutoTokenizer, AutoConfig

import logging
from pprint import pformat
import argparse
import os
import json
from pathlib import Path

from utils.switch import get_modules
from utils.auxiliary import set_seed
from utils.argument import verify_args

from tqdm import tqdm

logger = logging.getLogger(__file__)


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--params_file", type=str, help="JSON configuration file")
  parser.add_argument("--generate_config", type=str, help="JSON configuration file to generate")
  parser.add_argument("--dataset_path", type=str, default="data", help="Path of the dataset.")
  parser.add_argument("--dataset_cache", type=str, default='data/dataset_cache', help="Path of the dataset cache")
  parser.add_argument("--model_checkpoint", type=str, required=True, help="Path, url or short name of the model")
  parser.add_argument("--MAX_UTTERANCE_NUM", type=int, default=5, help="MAX_UTTERANCE_NUM")
  parser.add_argument("--MAX_SPEAKER_NUM", type=int, default=5, help="MAX_SPEAKER_NUM")
  parser.add_argument("--result_file", type=str, required=True, help="Path generate result")
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training (-1: not distributed)")
  parser.add_argument("--fp16", type=str, default="",
                      help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
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
    args.update(params)
    args = argparse.Namespace(**args)
  
  with open(args.generate_config, "r") as f:
    params = json.load(f)
    args = vars(args)
    args.update(params)
    args = argparse.Namespace(**args)

  dataloader, models, helper = get_modules(args)

  logger.info("Arguments: %s", pformat(args))

  args.n_gpu = 1
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args.device = device

  # Set seed
  set_seed(args)

  # Model construction
  config = AutoConfig.from_pretrained(args.model_name_or_path)
  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  try:
    SPECIAL_TOKENS = dataloader.SPECIAL_TOKENS
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
  except:
    pass
  args._tokenizer = tokenizer

  # Model construction
  model_class = getattr(models, args.model_class)
  if hasattr(args, "decoder_model_name_or_path"):
    from transformers.modeling_encoder_decoder import EncoderDecoderConfig
    config_encoder = config_decoder = config
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = model_class.from_encoder_decoder_pretrained(encoder_pretrained_model_name_or_path=args.model_name_or_path, 
          decoder_pretrained_model_name_or_path=args.decoder_model_name_or_path, config=config, args=args)
  else:
    if 'gpt' in args.model_class.lower():
      model = model_class.from_pretrained(args.model_name_or_path, config=config)
    else:
      model = model_class.from_pretrained(args.model_name_or_path, config=config, args=args)
  model.resize_token_embeddings(len(tokenizer))
  model = model.to(device)

  checkpoint_fp = Path(args.model_checkpoint)
  if checkpoint_fp.is_dir():
    checkpoint_fp = max(filter(lambda x: x.name.startswith("best_model"), checkpoint_fp.iterdir()), key=lambda x: float(x.stem.split('=')[-1]))
  assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
  logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
  checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
  Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
  
  dataset_class = getattr(dataloader, "testDataset") # TODO: suitable
  test_dataset = dataset_class(args=args, tokenizer=tokenizer, split_type='test', line_batch_list=[])
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn) #, collate_fn=test_dataset.collate_fn
  print("test dataset is ok ...")

  if args.debug:
    setattr(dataset_class, "__len__", lambda _: 20)

  model.eval()
  all_output_texts = []
  run_batch_generation_sample = helper.greedy_sample
  for did, batch in enumerate(tqdm(test_loader, desc="Generating", disable=args.debug)):
    with torch.no_grad():
      sampled_output_ids, ground_truth, history = run_batch_generation_sample(batch, args, model, test_dataset)
      if args.beam_search:
        sampled_output_ids = sampled_output_ids[0]
      sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)

      all_output_texts.append(sampled_output_text)
      if args.debug:
        print(f"Dialog: {did}")
        for i, h in enumerate(history):
          print(i, tokenizer.decode(h, skip_special_tokens=True))
        print("Generate:", sampled_output_text)
        print("  Ground:", ground_truth)
        print()

  with open(os.path.join("results", args.result_file), "w") as fout:
    for line in all_output_texts:
      fout.write(f"{line}\n")

if __name__ == '__main__':
  main()