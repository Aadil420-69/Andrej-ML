# pylint: disable=bad-indentation
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

"""
A Bigram Language Model which attempts to produce Shakespeare like stuff
Self-attention + Scaled
"""

# Importing stuff
from itertools import combinations
import random
random.seed(42)

from imp import Module as M
module = M()

from classes import DataClass, GPTConfig, GPT, estimate_loss, get_batch

import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(1337)

max_number = 10000


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
default_loc = "Practice/Addition/model.pth.tar"
res_number = len(str(max_number))

# Take user input
if input('non-default values? '):
	new_model = input('Do you want to train an existing model (default 1)? ')
	if new_model == '': new_model = 1

	if not new_model:
		get_loc = input('Where is the current model (Default "Practice/Addition/model.pth.tar")? ')
		if get_loc == '': get_loc = 'Practice/Addition/model.pth.tar'

	save_model = input('Do you want to save the trained model (default 1)?')
	if save_model == '': save_model = 1

	if save_model:
		save_loc = input('Where do you want to save it (Default "Practice/Addition/model.pth.tar")? ')
		if save_loc == '': save_loc = 'Practice/Addition/model.pth.tar'
else:
	new_model = 0
	save_model = 1
	get_loc = save_loc = 'Practice/Addition/model.pth.tar'

module.log('Creating Database', out=True)

# Creating database
nums = list(combinations(range(1, max_number), 2))
random.shuffle(nums)
module.log('Number of data: ' + str(len(nums)), out=True)


# Create mappings
text = list('0123456789+=$')
itos = dict(enumerate(text))
stoi = {s: i for i, s in enumerate(text)}
vocab_size = len(text)


# Create function to encode and decode
decode = lambda arr: ''.join(itos[char] for char in arr if itos[char] != '$')
e = lambda s: list(stoi[char] for char in s)

def encode(arr: list[int]) -> tuple[list[int], list[int]]:
	# arr.append(sum(arr))
	n = block_size - len(arr) - sum(len(str(num)) for num in arr)
	if n < 0:
		print('ERROR')
		print(arr, n)

	out = e('$')*n
	for char in arr:
		out += e(str(char) + '+')
	out = out[:-1] + e('=')

	summ = str(sum(arr))
	out2 = e('0'*(res_number - len(summ)) + summ)

	return (out, out2)

module.log('Encoding the Data', out=True)

# train test split
X = []
Y = []

# i = 0
n = len(nums)
for i, num in enumerate(nums):
	if i%10000 == 0:
		# print(i)
		module.log(f'Encoding {str(i)} |    {round(i / n * 100, 2)}%')
	arr, res = encode(num)
	for y in res:
		X.append(arr)
		Y.append(arr[1:] + [y])
		arr = arr[1:] + [y]
	# i+=1

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)


module.log('Saving Data', out=True)

# So that i wont have to make the db everytime
module.save_tensor(X, 'Practice/Addition/Data/input.txt')
module.save_tensor(Y, 'Practice/Addition/Data/result.txt')
with open('Practice/Addition/Data/input.txt', 'w') as f:
	f.write(str(X))
with open('Practice/Addition/Data/result.txt', 'w') as f:
	f.write(str(Y))


module.log('Setting up the Model and Optimizer', out=True)

# get the train/eval data and hyperparameter ready
data = DataClass(X, Y)
config = GPTConfig(data, vocab_size, max_number, res_number,
				   batch_size, block_size, max_iters,
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

	if iterr%10 == 0:
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


module.log('Generating from the Model')

# Generate
for i in range(20):
	r = [random.randint(1, max_number-1) for _ in range(2)]
	context = torch.tensor([encode(r)[0]], dtype=torch.long, device=config.device)
	out = decode(model.generate(context, max_new_tokens=5)[0].tolist())

	# print(i, out)
	module.log(str(i) + str(out), out=True)

module.end_log()

