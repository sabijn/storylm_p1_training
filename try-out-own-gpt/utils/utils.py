import tiktoken

def load_tokenizer(special_tokens):
	enc = tiktoken.get_encoding("cl100k_base")
	
	enc = tiktoken.Encoding(
		name="cl100k_custom",
		pat_str=enc._pat_str,
		mergeable_ranks=enc._mergeable_ranks,
		special_tokens={**enc._special_tokens, **special_tokens},
	)

	return enc