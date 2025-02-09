# pylint: disable=bad-indentation
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

"""
A Bigram Language Model which attempts to produce Shakespeare like stuff
After self-attention
"""

import torch
from torch import nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32	# how many independent processes will we process in parallel
block_size = 8	# whar is the maximum context length for prediction
max_iters = 5000
eval_intervals = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32


torch.manual_seed(1337)

with open('input.txt', 'r', encoding="utf-8") as f:
	text = f.read()

# get all the unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating a mapping for each char
stoi = { ch: i for i, ch in enumerate(chars) }
itos = dict(enumerate(chars))

# make funcs to encode nd decode
encode = lambda s: [stoi[c] for c in s]		# encoder: takes a string and outputs a list of integers
decode = lambda l: ''.join(itos[i] for i in l)	# decoder: takes a list of integers, outputs a string

# train test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
	"""_summary_
	Generates a small batch of data of inputs x and y

	Args:
		split (str): specifies whether to split from train set or test set

	Returns:
		torch.tensor[[int]]: the x and y component of the batch
	"""

	data = train_data if split == "train" else val_data
	ix = torch.randint(len(data) - batch_size, (batch_size, ))

	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])

	x, y = x.to(device), y.to(device)
	return x, y


@torch.no_grad()
def estimate_loss():
	"""_summary_
	Estimates the loss by calculating it several times and taking its mean

	Returns:
		dict[str: int]: a dict containing the mean losses of both the split modes
	"""
	out = {}
	model.eval()	# sets model into eval mode

	for split in ['train', 'eval']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			__, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()

	model.train()	# sets model into train mode
	return out

class Head(nn.Module):
	"""_summary_
	one head of self-attention
	"""

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
	
	def forward(self, x: torch.Tensor):
		"""_summary_
		forward passes the head

		Args:
			x (torch.Tensor): the activations of the different neurons on which the "affinity" has to be calculated

		Returns:
			torch.Tensor: the "affinities" of the neurons
		"""
		B, T, C = x.shape
		k: torch.Tensor = self.key(x)    # B, T, C
		q: torch.Tensor = self.query(x)    # B, T, C

		# compute attention scores ("affinities")
		wei: torch.Tensor = q @ k.transpose(-2, -1) * C**-0.5    # (B, T, C) @ (B, C, T) --> (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
		wei = F.softmax(wei, dim=-1)    # (B, T, T)

		# perform the weighted aggregation of the values
		v = self.value(x)    # (B, T, C)
		out = wei @ v    # (B, T, T) @ (B, T, C) --> (B, T, C)
		return out


class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)
	
	def forward(self, x):
		"""_summary_
		Args:
			x (torch.Tensor): the activations of the different neurons

		Returns:
			torch.Tensor: the "affinities" of the neurons after a multi-dimensional attention
		"""
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.proj(out)
		return out

class FeedForward(nn.Module):
	"""_summary_
	a simple linear layer followed by a non-linearity
	"""

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),    # projection
		)

	def forward(self, x: torch.Tensor):
		"""_summary_
		Args:
			x (torch.Tensor): the inputs/preactivation

		Returns:
			torch.Tensor: the activations of each neuron after a linear and non-linear layer each
		"""
		return self.net(x)


class Block(nn.Module):
	"""_summary_
	Transformer Block: communication followed by computation
	"""

	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)
	
	def forward(self, x: torch.Tensor):
		"""_summary_
		Args:
			x (torch.Tensor): the activations of the different neurons on which the "affinity" has to be calculated

		Returns:
			torch.Tensor: the "affinities" of the neurons
		"""
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x


class BigramLanguageModel(nn.Module):
	"""_summary_
	A class for BLM which is a child of nn.Module
	It has functions to forward and generate
	"""
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token in the lookup table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(
			Block(n_embd, n_head=4),
			Block(n_embd, n_head=4),
			Block(n_embd, n_head=4),
			nn.LayerNorm(n_embd),
		)
		# self.sa_heads = MultiHeadAttention(4, n_embd // 4)    # ie 4 heads of 8-dimensional self-attention
		# self.ffwd = FeedForward(n_embd)
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def forward(self, idx, targets=None):
		"""_summary_
		forward passes the model
		Args:
			idx (torch.tensor[int]): the indexes of the selected values of the batch
			targets (torch.tensor[int], optional): the indexes of the respected target values
			# idx and targets are both (B, T) tensors of integers

		Returns:
			torch.tensor[int]: the logits and loss
		"""
		B, T = idx.shape

		tok_emb = self.token_embedding_table(idx)    # (B, T, C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, C)
		x = tok_emb + pos_emb    # (B, T, C)
		# x = self.sa_heads(x)    # apply one head of self-attention, (B, T, C)
		# x = self.ffwd(x)    # (B, T, C)
		x = self.blocks(x)     # (B, T, C)
		logits = self.lm_head(x)    # (B, T, vocab_size)

		if targets is None:
			loss = None
		else:
			B,T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)

			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		"""_summary_

		Args:
			idx (torch.tensor[int]): the indexes of the selected values of a batch
			max_new_tokens (int): how many new characters should be generated

		Returns:
			torch.tensor[int]: the indexes of the chars of paragraph formed by the model
		"""
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop idx to the last block_size token
			idx_cond = idx[:, -block_size:]
			# get the Predicitons
			logits, __ = self(idx_cond)
			# focus only on the last T step
			logits = logits[:, -1, :]	# becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1)	# (B, C)
			# sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1)	# (B, 1)
			# append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1)	# (B, T+1)
		return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iterr in range(max_iters):
	# every once in a while calculate the loss on train and eval sets
	if iterr % eval_intervals == 0:
		losses = estimate_loss()
		print(f"step {iterr}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")

	# Sample a batch data
	xb, yb = get_batch('train')

	# evaluate the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()

losses = estimate_loss()
print(f"step {max_iters-1}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
