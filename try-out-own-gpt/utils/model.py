import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class Head(nn.Module):
	"""
	One head of self-attention
	"""
	def __init__(self, n_embd, head_size, block_size, dropout):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False) # what am I?
		self.query = nn.Linear(n_embd, head_size, bias=False) # what do I need?
		self.value = nn.Linear(n_embd, head_size, bias=False) # if you like me, this is what I will communicate
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # create a buffer 'non registered variable'

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape

		k = self.key(x)
		q = self.query(x)
		wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # don't transpose batch size, (B, T, 16) @ (B, 16, T) --> (B, T, T), so no communication across batches!

		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # if encoder blocks, then self-attention also in future
		wei = F.softmax(wei, dim=-1)
		wei = self.dropout(wei)
		v = self.value(x)
		out = wei @ v

		return out

class MultiHeadAttention(nn.Module):
	def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
		super().__init__()
		self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out
	
class FeedForward(nn.Module):

	def __init__(self, n_embd, dropout):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd), # residual projection
			nn.Dropout(dropout)
		)
	
	def forward(self, x):
		return self.net(x)

class Block(nn.Module):

	def __init__(self, n_embd, n_head, block_size, dropout):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
		self.ffwd = FeedForward(n_embd, dropout)
		self.ln1 = nn.LayerNorm(n_embd) # layer norm per token
		self.ln2 = nn.LayerNorm(n_embd) # layer norm per token
	
	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))

		return x

class GPTLanguageModel(nn.Module):

	def __init__(self, vocab_size, **config):
		super().__init__()
		self.block_size = config['block_size']
		self.token_embedding_table = nn.Embedding(vocab_size, config['n_embd'])
		self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embd'])
		self.blocks = nn.Sequential(*[Block(config['n_embd'], config['n_head'], config['block_size'], dropout=config['dropout']) for _ in range(config['n_layer'])])
		self.ln_f = nn.LayerNorm(config['n_embd'])
		self.lm_head = nn.Linear(config['n_embd'], vocab_size)
	
	def forward(self, idx, targets=None):
		B, T = idx.shape
		
		tok_emb = self.token_embedding_table(idx) # (Batch, Time (block_size), Channel)
		pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
		x = tok_emb + pos_emb
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.reshape(B*T, C)
			targets = targets.reshape(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss
	
	def generate(self, idx, max_new_tokens, stop_token_id=None):
		# idx is (B, T) array of indices
		B = idx.size(0)
		finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

		for _ in range(max_new_tokens):
			idx_cond = idx[:, -self.block_size:]
			logits, loss = self(idx_cond)
			logits = logits[:, -1, :]  # (B, vocab_size)
			probs = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

			if stop_token_id is not None:
				# once a sequence is finished, keep emitting stop_token_id
				idx_next[finished] = stop_token_id

				newly_finished = (idx_next.squeeze(-1) == stop_token_id)
				finished = finished | newly_finished

			idx = torch.cat((idx, idx_next), dim=1)

			if stop_token_id is not None and finished.all():
				break

		return idx