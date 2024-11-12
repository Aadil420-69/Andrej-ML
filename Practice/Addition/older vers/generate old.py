from imp import Module as M
module = M(log_file='generate.out')
import torch.nn as nn
from classes import GPT

import torch
torch.manual_seed(1337)

default_loc = "Practice/Addition/model.pth.tar"


with open('input.txt', 'r', encoding="utf-8") as f:
	text = f.read()
	chars = sorted(list(set(text)))
	itos = dict(enumerate(chars))
	decode = lambda l: ''.join(itos[i] for i in l)

get_loc = input(f'Where is the current model (Default "{default_loc}")? ')
if get_loc == '': get_loc = default_loc

state = module.load_model(get_loc)
config = state['config']
model = GPT(config)
model.load_state_dict(state['model'])
epoch = state['epoch']
print(model.numel())
module.log(f'Current epoch is {epoch}', out=True)


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
	n = config.block_size - len(arr) - sum(len(str(num)) for num in arr)
	if n < 0:
		print('ERROR')
		print(arr, n)

	out = e('$')*n
	for char in arr:
		out += e(str(char) + '+')
	out = out[:-1] + e('=')

	summ = str(sum(arr))
	out2 = e('0'*(5 - len(summ)) + summ)

	return (out, out2)

import random

# generate from the model
for i in range(20):
	nums = [random.randint(1, 9999) for _ in range(2)]
	context = torch.tensor([encode(nums)[0]], dtype=torch.long, device=config.device)
	# context = torch.randint(0, 10000, (1, 2), dtype=torch.long, device=config.device)
	out = decode(model.generate(context, max_new_tokens=5)[0].tolist())

	# print(i, out)
	module.log(str(i) + str(out), out=True)
module.end_log()
