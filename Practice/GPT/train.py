# pylint: disable=bad-indentation
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

"""
A Bigram Language Model which attempts to produce Shakespeare like stuff
Self-attention + Scaled
"""
from imp import Module as M
module = M()

from classes import DataClass, GPTConfig, GPT, estimate_loss, get_batch

import torch
torch.manual_seed(1337)


# hyperparameters
batch_size = 16    # how many independent processes will we process in parallel
block_size = 256    # whar is the maximum context length for prediction
max_iters = 5000
eval_intervals = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# user vars
if input('non-default values? '):
	new_model = input('Do you want to train an existing model (default 1)? ')
	if new_model is None: new_model = 1

	if not new_model:
		get_loc = input('Where is the current model (Default "Practice/GPT/model.pth.tar")? ')
		if get_loc is None: get_loc = 'Practice/GPT/model.pth.tar'

	save_model = input('Do you want to save the trained model (default 1)?')
	if save_model is None: save_model = 1

	if save_model:
		save_loc = input('Where do you want to save it (Default "Practice/GPT/model.pth.tar")? ')
		if save_loc is None: save_loc = 'Practice/GPT/model.pth.tar'
else:
	new_model = 0
	save_model = 1
	get_loc = save_loc = 'Practice/GPT/model.pth.tar'

with open('input.txt', 'r', encoding="utf-8") as f:
	text = f.read()

# get all the unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating a mapping for each char
stoi = { ch: i for i, ch in enumerate(chars) }
itos = dict(enumerate(chars))

# make funcs to encode nd decode
encode = lambda s: [stoi[c] for c in s]		    # encoder: takes a string and outputs a list of integers
decode = lambda l: ''.join(itos[i] for i in l)	    # decoder: takes a list of integers, outputs a string


# get the train/eval data and hyperparameter ready
data = DataClass(encode(text))
config = GPTConfig(data, vocab_size, batch_size, block_size,
				   max_iters, eval_intervals, learning_rate,
				   eval_intervals, n_embd, n_head,
				   n_layer, dropout, False)

# cerate the Model
model = GPT(config)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

epoch = 0

# get existing stuff
if not new_model:
	state = module.load_model(get_loc)
	model.load_state_dict(state['model'])
	optimizer.load_state_dict(state['model'])
	epoch = state['epoch']
	config = state['config']


# iterr = 350

for iterr in range(max_iters):
	with open('Practice/GPT/exit.txt', 'r') as f:
		if f.read():
			module.log('ending training', out=True)
			break

	module.log(f'iteration {iterr}', out=True)
	# every once in a while calculate the loss on train and eval sets
	if iterr % eval_intervals == 0:
		losses = estimate_loss(model, config)
		module.log(f"step {iterr}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}", out=True)

		# saves the model
		if save_model:
			state = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch+iterr,
				'config': config
			}
			module.save_model(state, save_loc)    # saves the model

	# Sample a batch data
	xb, yb = get_batch('train', config)

	# evaluate the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
	# break

losses = estimate_loss(model, config)
module.log(f"step {max_iters-1}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}", out=True)

# saves the model
if save_model:
	state = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch+iterr,
		'config': config
	}
	module.save_model(state, save_loc)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

module.end_log()
