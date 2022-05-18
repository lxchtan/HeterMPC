import torch
import torch.nn.functional as F
import dgl
import numpy as np
from ignite.utils import convert_tensor


from utils.auxiliary import top_filtering
from copy import deepcopy

train_metrics = ["loss", "lm"]

def trainer_update(engine, batch, args, model, optimizer):
  """
    engine.state.epoch: start from 1
    engine.state.iteration: continue after epochs, not set back 0
    engine.state.epoch_length
  """
  model.train()
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
  graph, input_ids, decode_input, segment_id, input_masks, lm_labels, ans_idxs, ans_from = batch
  outputs = model(graph=graph, input_ids=input_ids, decoder_input_ids=decode_input, token_type_ids=None, labels=lm_labels, attention_mask=input_masks, decoder_ans_idxs=ans_idxs, decoder_ans_from=ans_from, return_dict=True)

  lm_loss = outputs.loss
  loss = lm_loss

  return_output = (loss.item(), lm_loss.item())

  if args.n_gpu > 1:
    loss = loss.mean()
  if args.gradient_accumulation_steps > 1:
    loss = loss / args.gradient_accumulation_steps

  loss.backward()
  if args.max_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
  if engine.state.iteration % args.gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
  return return_output


def evaluator_update(engine, batch, args, model):
  model.eval()
  with torch.no_grad():
    # batch = tuple(convert_tensor(input_tensor, device=args.device, non_blocking=True) for input_tensor in batch)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    graph, input_ids, decode_input, segment_id, input_masks, lm_labels, ans_idxs, ans_from = batch
    outputs = model(graph=graph, input_ids=input_ids, decoder_input_ids=decode_input, token_type_ids=None, labels=lm_labels, attention_mask=input_masks, decoder_ans_idxs=ans_idxs, decoder_ans_from=ans_from, return_dict=True)

    lm_logits = outputs.logits

  return (lm_logits.view(-1, lm_logits.size(-1)), ), (lm_labels.view(-1), )

def greedy_sample(batch, args, model, dataset):
  bos_id = args._tokenizer.bos_token_id
  eos_id = args._tokenizer.eos_token_id
  pad_id = args._tokenizer.pad_token_id

  num_beams = args.beam_size if args.beam_search else 1

  graph = batch[0].to(args.device)
  input_ids, input_mask, ans_idxs, ans_from = tuple(convert_tensor(input_tensor, device=args.device, non_blocking=True) for input_tensor in batch[1:5])
  history, response_text = batch[5]
  
  current_outputs = model.generate(
      graph=graph,
      input_ids=input_ids,
      attention_mask=input_mask,
      decoder_ans_idxs=ans_idxs,
      decoder_ans_from=ans_from,  
      max_length=args.max_length,
      min_length=args.min_length,
      do_sample=not args.no_sample,
      num_beams=num_beams,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      # repetition_penalty=1.2,
      # bad_words_ids: Optional[Iterable[int]] = None,
      bos_token_id=bos_id,
      pad_token_id=pad_id,
      eos_token_id=eos_id,
      # length_penalty: Optional[float] = None,
      # no_repeat_ngram_size=1,
      num_return_sequences=1,
      decoder_start_token_id=bos_id,  # To ensure start with bos
      use_cache=False,  # Now the code does not yet support generate using cache, change this to avoid caching
  )

  return current_outputs[0], response_text[0], history[0]

