import torch
import torch.nn.functional as F

import argparse
from pathlib import Path
import os
import re
import csv
import yaml

from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model import GPTLanguageModel
from utils import load_tokenizer
from sklearn.metrics import f1_score


@dataclass
class XLWICExample:
	word: str
	sentence1: str
	sentence2: str
	postag: str
	pos1: tuple[int, int]
	pos2: tuple[int, int]
	label: int


def load_model(vocab_size, device, config):
	model = GPTLanguageModel(vocab_size, **config["model"])

	state_dict = torch.load(config["paths"]["output_model_path"], map_location=device)
	model.load_state_dict(state_dict)

	model.to(device)
	model.eval()
	return model


def sanitize_filename(name: str) -> str:
	return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


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


def render_prompt(template: str, example: XLWICExample) -> str:
	return (
		template.replace("{{ target_word }}", example.word)
		.replace("{{ example_1 }}", example.sentence1)
		.replace("{{ example_2 }}", example.sentence2)
	)


@torch.no_grad()
def continuation_logprob(model, tokenizer, device, prompt: str, continuation: str, normalize=False) -> float:
	prompt_ids = encode_text(tokenizer, prompt).to(device)
	full_ids = encode_text(tokenizer, prompt + continuation).to(device)

	prompt_len = prompt_ids.size(1)
	full_len = full_ids.size(1)

	logits, _ = model(full_ids)
	log_probs = F.log_softmax(logits, dim=-1)

	total_logprob = 0.0
	num_tokens = full_len - prompt_len

	for pos in range(prompt_len, full_len):
		token_id = full_ids[0, pos]
		total_logprob += log_probs[0, pos - 1, token_id].item()

	if normalize:
		return total_logprob / num_tokens
	return total_logprob


@torch.no_grad()
def predict_example(model, tokenizer, device, prompt: str):
	"""
	Label mapping:
	  1 -> identiek
	  0 -> verschillend
	"""
	option_identiek = " identiek"
	option_verschillend = " verschillend"

	lp_identiek = continuation_logprob(model, tokenizer, device, prompt, option_identiek)
	lp_verschillend = continuation_logprob(model, tokenizer, device, prompt, option_verschillend)

	pred_label = 1 if lp_identiek > lp_verschillend else 0

	return {
		"pred_label": pred_label,
		"logprob_identiek": lp_identiek,
		"logprob_verschillend": lp_verschillend,
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--result_path",
		type=Path,
		default=Path("/local/perdijks/training_gpt/xlwic_results"),
	)
	args = parser.parse_args()

	xlwic_path = Path("/local/perdijks/datasets/dutch_nl/nl_test_data.txt")
	xlwic_gold_path = Path("/local/perdijks/datasets/dutch_nl/nl_test_gold.txt")
	xlwic_valid_path = Path("/local/perdijks/datasets/dutch_nl/nl_valid.txt")  # kept for future use

	with open("config.yaml", "r") as f:
		config = yaml.safe_load(f)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_name = Path(config["paths"]["output_model_path"]).stem

	# Define special tokens
	EOS = config["tokens"]["eos_token"]
	BOS = config["tokens"]["bos_token"]

	# Ensure special tokens exist in encoding
	special_tokens = {
		BOS: 100264,
		EOS: 100257,
	}

	tokenizer = load_tokenizer(special_tokens)
	model = load_model(tokenizer.n_vocab, device, config)
	model.eval()

	results_dir = Path(args.result_path)
	results_dir.mkdir(parents=True, exist_ok=True)

	with open(xlwic_gold_path, encoding="utf8") as gold_file:
		gold_labels = [int(line.strip()) for line in gold_file.readlines()]

	dataset = []
	with open(xlwic_path, encoding="utf8") as truth_file:
		for i, line in enumerate(truth_file.readlines()):
			example = line.strip().split("\t")
			dataset.append(
				XLWICExample(
					word=example[0],
					sentence1=example[6],
					sentence2=example[7],
					postag=example[1],
					pos1=(int(example[2]), int(example[3])),
					pos2=(int(example[4]), int(example[5])),
					label=gold_labels[i],
				)
			)

	prompt_template = """Is de betekenis van ’{{ target_word }}’ in de volgende zinnen identiek of verschillend?
Zin 1: {{ example_1 }}
Zin 2: {{ example_2 }}
Antwoord met ’identiek’ of ’verschillend’. De betekenis van ’{{ target_word }}’ is"""

	all_rows = []
	correct = 0

	for i, example in enumerate(dataset):
		prompt = render_prompt(prompt_template, example)
		pred = predict_example(model, tokenizer, device, prompt)

		is_correct = int(pred["pred_label"] == example.label)
		correct += is_correct

		all_rows.append(
			{
				"idx": i,
				"word": example.word,
				"postag": example.postag,
				"sentence1": example.sentence1,
				"sentence2": example.sentence2,
				"gold_label": example.label,
				"pred_label": pred["pred_label"],
				"correct": is_correct,
				"logprob_identiek": pred["logprob_identiek"],
				"logprob_verschillend": pred["logprob_verschillend"],
			}
		)

		if (i + 1) % 100 == 0 or (i + 1) == len(dataset):
			acc = correct / (i + 1)
			print(f"[{i+1}/{len(dataset)}] accuracy={acc:.4f}")

	golds = [row["gold_label"] for row in all_rows]
	preds = [row["pred_label"] for row in all_rows]

	weighted_f1 = f1_score(golds, preds, average="weighted")
	accuracy = correct / len(dataset) if dataset else 0.0

	safe_model_name = sanitize_filename(model_name)
	pred_file = results_dir / f"xlwic_dutch_{safe_model_name}_predictions.csv"
	summary_file = results_dir / f"xlwic_dutch_{safe_model_name}_summary.txt"

	with open(pred_file, "w", encoding="utf8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"idx",
				"word",
				"postag",
				"sentence1",
				"sentence2",
				"gold_label",
				"pred_label",
				"correct",
				"logprob_identiek",
				"logprob_verschillend",
			],
		)
		writer.writeheader()
		writer.writerows(all_rows)

	with open(summary_file, "w", encoding="utf8") as f:
		f.write(f"model_name: {model_name}\n")
		f.write(f"dataset: xlwic_dutch_test\n")
		f.write(f"num_examples: {len(dataset)}\n")
		f.write(f"accuracy: {accuracy:.6f}\n")
		f.write(f"weighted_f1: {weighted_f1:.6f}\n")
		f.write(f"predictions_file: {pred_file}\n")

	print("Finished evaluation.")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Weighted F1: {weighted_f1:.4f}")
	print(f"Predictions saved to: {pred_file}")
	print(f"Summary saved to: {summary_file}")