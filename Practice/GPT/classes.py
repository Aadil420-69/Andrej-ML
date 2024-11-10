from imp import Module as M
module = M()

import torch
from torch import nn
import torch.nn.functional as F


# data loading
def get_batch(split: str, config: 'GPTConfig', iterr: int = -10):
	"""_summary_
	Generates a small batch of data of inputs x and y

	Args:
		split (str): specifies whether to split from train set or test set

	Returns:
		torch.tensor[[int]]: the x and y component of the batch
	"""
	data = config.data.train_data if split == "train" else config.data.val_data
	ix = torch.randint(len(data) - config.block_size, (config.batch_size, ))

	f = open('Practice/GPT/Logs/log.txt', 'w')
	if iterr >= 0: f.write('Evaluation loop number ' + str(iterr) + '\n')
		# print(iterr)
	f.write(str(module.time()) + '\n')
	f.write(str(module.time_since()) + 'secs\n')
	f.write(str(ix) + '\n')

	x = torch.stack([data[i:i+config.block_size] for i in ix])
	y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])

	f.write(str(x) + "\n")
	f.write(str(y) + "\n")
	f.close()

	x, y = x.to(config.device), y.to(config.device)
	return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, config: 'GPTConfig'):
	"""_summary_
	Estimates the loss by calculating it several times and taking its mean

	Returns:
		dict[str: int]: a dict containing the mean losses of both the split modes
	"""
	out = {}
	model.eval()	# sets model into eval mode
	i = 0

	for split in ['train', 'eval']:
		losses = torch.zeros(config.eval_iters)
		for k in range(config.eval_iters):
			X, Y = get_batch(split, config, i)
			__, loss = model(X, Y)
			losses[k] = loss.item()
			i += 1
		out[split] = losses.mean()

	model.train()	# sets model into train mode
	return out

class MultiHeadAttention(nn.Module):
	def __init__(self, config: 'GPTConfig'):
		# n_head, head_size
		super().__init__()
		self.config = config

		self.attn = nn.Linear(config.n_embd, config.n_embd*3, bias=config.bias)

		self.register_buffer('tril', torch.tril(torch.ones((config.block_size, config.block_size))))

		self.attn_dropout = nn.Dropout(config.dropout)
		self.proj = nn.Linear(config.n_embd, config.n_embd)
		self.resid_dropout = nn.Dropout(config.dropout)

	
	def forward(self, x: torch.Tensor):
		B, T, C = x.shape
		config = self.config

		# k = self.key(x).view(B, T, config.n_head, C // config.n_head).transpose(1, 2)	# (B, nh, T, hs)
		# q = self.query(x).view(B, T, config.n_head, C // config.n_head).transpose(1, 2)	# (B, nh, T, hs)
		# v = self.value(x).view(B, T, config.n_head, C // config.n_head).transpose(1, 2)	# (B, nh, T, hs)
		k, q, v = self.attn(x).split(config.n_embd, dim=2)
		k = k.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)	# (B, nh, T, hs)
		q = q.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)	# (B, nh, T, hs)
		v = v.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)	# (B, nh, T, hs)

		# compute attention scores ("affinities")
		wei: torch.Tensor = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5	# (B, nh, T, hs) @ (B, nh, T, hs) --> (B, nh, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))	# (B, nh, T, T)
		wei = F.softmax(wei, dim=-1)	# (B, nh, T, T)
		wei = self.attn_dropout(wei)	# (B, nh, T, T)

		# perform the weighted aggregation of the values
		out: torch.tensor = wei @ v	# (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
		out = out.transpose(1, 2).contiguous().view(B, T, C)	# (B, T, C)
		out = self.resid_dropout(self.proj(out))

		return out


class FeedForward(nn.Module):
	"""_summary_
	a simple linear layer followed by a non-linearity
	"""

	def __init__(self, config: 'GPTConfig'):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(config.n_embd, 4 * config.n_embd),
			nn.ReLU(),
			nn.Linear(4 * config.n_embd, config.n_embd),	# projection
			nn.Dropout(config.dropout),
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

	def __init__(self, config: 'GPTConfig'):
		# n_embd, n_head
		super().__init__()
		self.sa = MultiHeadAttention(config)
		self.ffwd = FeedForward(config)
		self.ln1 = nn.LayerNorm(config.n_embd)
		self.ln2 = nn.LayerNorm(config.n_embd)
	
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


class GPT(nn.Module):
	"""_summary_
	A class for BLM which is a child of nn.Module
	It has functions to forward and generate
	"""
	def __init__(self, config: 'GPTConfig' = None):
		super().__init__()
		if config:
			self.config = config

			# each token directly reads off the logits for the next token in the lookup table
			self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
			self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
			self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
			self.ln_f = nn.LayerNorm(config.n_embd)	# final layer norm
			self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

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
		config = self.config

		tok_emb = self.token_embedding_table(idx)	# (B, T, C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))	# (T, C)
		x = tok_emb + pos_emb	# (B, T, C)
		x = self.blocks(x)	 # (B, T, C)
		x = self.ln_f(x)	 # (B, T, C)
		logits = self.lm_head(x)	# (B, T, vocab_size)

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
		config = self.config

		for _ in range(max_new_tokens):
			# crop idx to the last block_size token
			idx_cond = idx[:, -config.block_size:]
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

	def numel(self, only_trainable: bool = True):
		"""
		Returns the total number of parameters (only counting
		shared parameters once); if `only_trainable` is True, then only
		includes parameters with `requires_grad = True`
		"""
		parameters = list(self.parameters())
		if only_trainable:
			parameters = [p for p in parameters if p.requires_grad]
		unique = {p.data_ptr(): p for p in parameters}.values()
		return sum(p.numel() for p in unique)


# hyperparameters
class GPTConfig:
	def __init__(self, data: 'DataClass', vocab_size: int, batch_size=64,
			  block_size: int = 256, max_iters: int = 5000,
			  eval_intervals: int = 500, learning_rate: float = 3e-4,
			  eval_iters: int = 200, n_embd: int = 384,
			  n_head: int = 6, n_layer: int = 6,
			  dropout: int = 0.2, bias: bool = False
			):
		self.data = data
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.block_size = block_size
		self.max_iters = max_iters
		self.eval_intervals = eval_intervals
		self.learning_rate = learning_rate
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.eval_iters = eval_iters
		self.n_embd = n_embd
		self.n_head = n_head
		self.n_layer = n_layer
		self.dropout = dropout
		self.head_size = n_embd / n_head
		self.bias = bias

class DataClass:
	def __init__(self, data: torch.Tensor, split: int = 0.9):
		data = torch.tensor(data, dtype=torch.long)
		n = int(split * len(data))
		self.train_data = data[:n]
		self.val_data = data[n:]

# # default parameters
# batch_size = 16	# how many independent processes will we process in parallel
# block_size = 256	# whar is the maximum context length for prediction
# max_iters = 20_000
# eval_intervals = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 50
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
