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


def load_model(vocab_size, model_path, device, model_config):
	print(config)
	model = GPTLanguageModel(vocab_size, **model_config)

	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict)

	model.to(device)
	model.eval()

	return model


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_model_path', type=Path, default='/local/perdijks/training_gpt/trained_models/gpt_try_out_storiesforthewin_chiscor.pt')

	args = parser.parse_args()

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	with open("config.yaml", "r") as f:
		config = yaml.safe_load(f)

	# Define special tokens
	EOS = config['tokens']['eos_token']
	BOS = config['tokens']['bos_token']

	# Ensure special tokens exist in encoding
	special_tokens = {
		BOS: 100264,      # typical ID for begin_of_text
		EOS: 100257       # typical ID for endoftext
	}
	tokenizer = load_tokenizer(special_tokens)

	model = load_model(tokenizer.n_vocab, args.output_model_path, device, config['model'])

	while True:
		prompt = input("\nPrompt: ")

		if prompt.strip() == "":
			break

		context = torch.tensor(
			[tokenizer.encode(prompt)],
			dtype=torch.long,
			device=device
		)

		generated = model.generate(context, config['inference']['max_new_tokens'], stop_token_id=special_tokens[EOS])
		new_tokens = generated[0][context.shape[1]:]
		output = tokenizer.decode(new_tokens.tolist())

		print("\n=== Output ===")
		print(output)