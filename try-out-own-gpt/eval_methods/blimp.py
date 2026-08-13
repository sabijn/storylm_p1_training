import torch
import torch.nn.functional as F

import argparse
import tiktoken

from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from model import GPTLanguageModel
from utils import load_tokenizer
import yaml


import re
import csv
import math
import pandas as pd
from datasets import load_dataset, get_dataset_config_names


def load_model(vocab_size, device, config, args):
	model = GPTLanguageModel(vocab_size, **config['model'])

	state_dict = torch.load(args.model, map_location=device)
	model.load_state_dict(state_dict)

	model.to(device)
	model.eval()

	return model

def sanitize_filename(name: str) -> str:
	return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def get_dataset_iter(ds):
	"""
	Returns an iterable over examples, whether ds is a Dataset or a DatasetDict.
	BLiMP subsets often come as a DatasetDict with a single split.
	"""
	if isinstance(ds, dict):
		# e.g. DatasetDict({"train": Dataset(...)})
		if "train" in ds:
			return ds["train"]
		# fallback: first split
		first_key = next(iter(ds.keys()))
		return ds[first_key]
	return ds


def encode_text(tokenizer, text):
	"""
	Tries HuggingFace-style tokenization first, then falls back to a custom tokenizer.encode().
	Returns a 1 x T LongTensor.
	"""
	if hasattr(tokenizer, "__call__"):
		try:
			encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
			if isinstance(encoded, dict) and "input_ids" in encoded:
				return encoded["input_ids"]
		except TypeError:
			pass

	if hasattr(tokenizer, "encode"):
		ids = tokenizer.encode(text)
		if isinstance(ids, torch.Tensor):
			if ids.dim() == 1:
				ids = ids.unsqueeze(0)
			return ids.long()
		return torch.tensor([ids], dtype=torch.long)

	raise ValueError("Tokenizer must support either __call__(..., return_tensors='pt') or .encode(text)")


@torch.no_grad()
def sentence_logprob(model, tokenizer, sentence, device, normalize_by_length=True):
	"""
	Computes sentence log-probability under a causal language model.

	For CLMs:
	  p(x1, ..., xT) = product_t p(x_t | x_<t)

	We compute the sum of token log-probs for positions 1..T-1
	using shifted logits and shifted labels.

	normalize_by_length=True uses average log-prob per predicted token,
	which is standard for BLiMP-style minimal pair evaluation to reduce
	length bias.
	"""
	input_ids = encode_text(tokenizer, sentence).to(device)

	outputs = model(input_ids)
	logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

	# Shift for next-token prediction
	shift_logits = logits[:, :-1, :].contiguous()
	shift_labels = input_ids[:, 1:].contiguous()

	log_probs = torch.log_softmax(shift_logits, dim=-1)
	token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

	total_logprob = token_log_probs.sum().item()
	num_pred_tokens = shift_labels.numel()

	if normalize_by_length and num_pred_tokens > 0:
		return total_logprob / num_pred_tokens
	return total_logprob


def evaluate_blimp_subset(model, tokenizer, subset_name, subset_data, device, normalize_by_length=True):
	ds = get_dataset_iter(subset_data)

	rows = []
	correct = 0
	total = 0

	for idx, example in enumerate(ds):
		good_sent = example["sentence_good"]
		bad_sent = example["sentence_bad"]
		phenomenon = example.get("linguistic_phenomenon", subset_name)

		good_score = sentence_logprob(
			model=model,
			tokenizer=tokenizer,
			sentence=good_sent,
			device=device,
			normalize_by_length=normalize_by_length,
		)

		bad_score = sentence_logprob(
			model=model,
			tokenizer=tokenizer,
			sentence=bad_sent,
			device=device,
			normalize_by_length=normalize_by_length,
		)

		is_correct = good_score > bad_score
		correct += int(is_correct)
		total += 1

		rows.append({
			"subset": subset_name,
			"index": idx,
			"linguistic_phenomenon": phenomenon,
			"sentence_good": good_sent,
			"sentence_bad": bad_sent,
			"good_score": good_score,
			"bad_score": bad_score,
			"margin": good_score - bad_score,
			"correct": int(is_correct),
		})

	accuracy = correct / total if total > 0 else float("nan")
	return rows, {
		"subset": subset_name,
		"n_examples": total,
		"n_correct": correct,
		"accuracy": accuracy,
	}


def evaluate_blimp_nl(model_name, model, tokenizer, device, output_dir, normalize_by_length=True):

	subset_names = get_dataset_config_names("juletxara/blimp-nl")
	blimp_nl = {
		subset: load_dataset("juletxara/blimp-nl", subset)
		for subset in subset_names
	}

	summary_rows = []

	for subset_name, subset_data in blimp_nl.items():
		print(f"Evaluating subset: {subset_name}")

		detailed_rows, summary = evaluate_blimp_subset(
			model=model,
			tokenizer=tokenizer,
			subset_name=subset_name,
			subset_data=subset_data,
			device=device,
			normalize_by_length=normalize_by_length,
		)

		summary_rows.append(summary)

		subset_filename = sanitize_filename(subset_name)
		subset_csv_path = os.path.join(output_dir, f"{model_name}_{subset_filename}.csv")
		pd.DataFrame(detailed_rows).to_csv(subset_csv_path, index=False)

		print(
			f"  -> accuracy = {summary['accuracy']:.4f} "
			f"({summary['n_correct']}/{summary['n_examples']})"
		)
		print(f"  -> saved detailed results to {subset_csv_path}")

	summary_df = pd.DataFrame(summary_rows).sort_values("subset")
	summary_csv_path = os.path.join(output_dir, f"{model_name}_blimp_nl_summary.csv")
	summary_df.to_csv(summary_csv_path, index=False)

	# Also save macro average
	macro_acc = summary_df["accuracy"].mean()
	with open(os.path.join(output_dir, f"{model_name}_blimp_nl_macro_accuracy.txt"), "w") as f:
		f.write(f"{macro_acc:.6f}\n")

	print(f"\nSaved summary to {summary_csv_path}")
	print(f"Macro-average accuracy: {macro_acc:.4f}")

	return summary_df

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--result_path', type=Path, default='/local/perdijks/training_gpt/blimp_results')
	parser.add_argument('--model', type=str, required=True)
	args = parser.parse_args()

	with open("config.yaml", "r") as f:
		config = yaml.safe_load(f)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model_name = args.result_path.name
	
	# Define special tokens
	EOS = config['tokens']['eos_token']
	BOS = config['tokens']['bos_token']

	# Ensure special tokens exist in encoding
	special_tokens = {
		BOS: 100264,      # typical ID for begin_of_text
		EOS: 100257       # typical ID for endoftext
	}

	tokenizer = load_tokenizer(special_tokens)
	model = load_model(tokenizer.n_vocab, device, config, args)
	model.eval()

	summary_df = evaluate_blimp_nl(
		model_name=model_name,
		model=model,
		tokenizer=tokenizer,
		device=device,
		output_dir=args.result_path,
		normalize_by_length=True,
	)

	print(summary_df)
