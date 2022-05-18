import torch
from ignite.utils import convert_tensor

train_metrics = ["loss", "lm"]

def trainer_update(engine, batch, args, model, optimizer):
  """
    engine.state.epoch: start from 1
    engine.state.iteration: continue after epochs, not set back 0
    engine.state.epoch_length
  """
  model.train()
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
  graph, input_ids, decode_input, segment_id, input_masks, decoder_attention_mask, lm_labels, ans_idxs = batch
  outputs = model(graph=graph, input_ids=input_ids, decoder_input_ids=decode_input, decoder_attention_mask=decoder_attention_mask, token_type_ids=None, labels=lm_labels, attention_mask=input_masks, ans_idxs=ans_idxs)

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
    graph, input_ids, decode_input, segment_id, input_masks, decoder_attention_mask, lm_labels, ans_idxs = batch
    outputs = model(graph=graph, input_ids=input_ids, decoder_input_ids=decode_input, decoder_attention_mask=decoder_attention_mask, token_type_ids=None, attention_mask=input_masks, ans_idxs=ans_idxs)

    lm_logits = outputs.logits

    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
    lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

  return (lm_logits_flat_shifted, ), (lm_labels_flat_shifted, )

def greedy_sample(batch, args, model, dataset):
  bos_id = args._tokenizer.convert_tokens_to_ids("[CLS]")
  eos_id = args._tokenizer.convert_tokens_to_ids("[SEP]")
  pad_id = args._tokenizer.convert_tokens_to_ids("[PAD]")

  # num_beams = args.sub_beam_size * args.group_num
  num_beams = 1

  example = batch[0]  # 'history', 'response', 'response_text', 'emotion_label', 'label', 'dialog_id'

  graph, history = example["graph"], example["history"]
  response_text = example["response_text"]
  # ans_spk = example['answer_spk']

  ins = dataset.build_graph_from_examples(graph, history, [], with_eos=False)

  current_output = model.generate(
      input_ids=ins['input_datas']['encoding'].to(args.device),
      graph=graph.to(args.device),
      attention_mask=ins['input_datas']['encoding_mask'].to(args.device),
      ans_idxs=torch.tensor(ins['answer_index'], device=args.device).unsqueeze(0),
      max_length=args.max_length,
      min_length=args.min_length,
      do_sample=not args.no_sample,
      # do_sample=True,
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
      num_return_sequences=num_beams,
      decoder_start_token_id=bos_id,
      # use_cache: Optional[bool] = None,
  )[0]

  return current_output, response_text, history
