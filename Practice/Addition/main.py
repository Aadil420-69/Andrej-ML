# pylint: disable=bad-indentation
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

"""
A Bigram Language Model which attempts to produce Shakespeare like stuff
Self-attention + Scaled
"""

# Importing stuff
import random
random.seed(42)

from libs.classes import GPTConfig, GPT, estimate_loss, get_batch
from libs.data import DataClass
from libs.imp import Module as M
module = M()

import torch
torch.manual_seed(1337)

max_number = 10000

with open('Practice/Addition/exit.txt', 'r') as f:
	if f.read():
		with open('Practice/Addition/exit.txt', 'w') as f:
			module.log('Clearing exit.txt', out=True)


# hyperparameters
batch_size = 64
block_size = 16
max_iters = 1_000_000
eval_intervals = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 96
n_head = 6
n_layer = 6
dropout = 0.2
default_loc = "Practice/Addition/Models/model.pth.tar"
res_number = len(str(max_number))

# Take user input
if input('non-default values? '):
	new_model = input('Do you want to train an existing model (default 1)? ')
	if new_model == '': new_model = 1

	if not new_model:
		get_loc = input(f'Where is the current model (Default {default_loc})? ')
		if get_loc == '': get_loc = default_loc

	save_model = input('Do you want to save the trained model (default 1)?')
	if save_model == '': save_model = 1

	if save_model:
		save_loc = input(f'Where do you want to save it (Default {default_loc})? ')
		if save_loc == '': save_loc = default_loc
else:
	new_model = 0
	save_model = 1
	get_loc = default_loc
	save_loc = default_loc

module.log('Fetching Database', out=True)

# Creating database
# from itertools import combinations
# nums = list(combinations(range(1, max_number), 2))
# random.shuffle(nums)
# module.log('Number of data: ' + str(len(nums)), out=True)

# module.save_tensor(torch.tensor(nums, dtype=torch.long), 'Practice/Addition/Data/input.txt')
nums = module.load_tensor('Practice/Addition/Data/input.txt')
data = DataClass(nums, res_number, block_size)

module.log('Setting up the Model and Optimizer', out=True)

# get the train/eval data and hyperparameter ready
config = GPTConfig(data, max_number, batch_size, max_iters,
				   eval_intervals, learning_rate, eval_iters,
				   n_embd, n_head, n_layer, dropout, False)


# cerate the Model
model = GPT(config)
model = model.to(device)


# create a PyTorch optimizer and define epoch
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
epoch = 0


# get existing stuff
if not new_model:
	state = module.load_model(get_loc)
	model.load_state_dict(state['model'])
	optimizer.load_state_dict(state['optimizer'])
	epoch = state['epoch']
	config = state['config']


module.log('Starting Training', out=True)

eval_bool = False
b = False
module.log(epoch)
for iterr in range(max_iters):
	with open('Practice/Addition/exit.txt', 'r') as f:
		if f.read():
			module.log('ending training', out=True)
			epoch -= 1
			b = True
	if b: break

	if iterr%50 == 0:
		module.log(f'iteration {iterr}', out=True)

	# every once in a while calculate the loss on train and eval sets
	if iterr % eval_intervals == 0:
		if not eval_bool:
			eval_bool = True
		else:
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
				module.save_model(state, save_loc)	# saves the model

	# Sample a batch data
	xb, yb = get_batch('train', config)
	# assert 0 > 1

	# evaluate the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
	# break

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
	module.save_model(state, save_loc)


module.log('Generating from the Model')

# Generate
for i in range(1, 21):
	r = [random.randint(1, max_number-1) for _ in range(2)]
	context = config.data.encode(r, 0)[0].reshape(1, config.data.block_size)
	# print(config.data.encode(r, 0))
	out = data.decode(model.generate_mul(context, max_new_tokens=config.data.res_number)[0].tolist())
	module.log(f'{i}: {out}', out=True)
	# module.log(str(i) + str(out), out=True)


module.end_log()

