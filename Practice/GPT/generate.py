from imp import Module as M
module = M(log_file='generate.out')
import torch.nn as nn
from classes import GPT

import torch
torch.manual_seed(1337)

with open('input.txt', 'r', encoding="utf-8") as f:
	text = f.read()
	chars = sorted(list(set(text)))
	itos = dict(enumerate(chars))
	decode = lambda l: ''.join(itos[i] for i in l)

get_loc = input('Where is the current model (Default "Practice/GPT/model.pth.tar")? ')
if get_loc == '': get_loc = 'Practice/GPT/model.pth.tar'

state = module.load_model(get_loc)
config = state['config']
model = GPT(config)
model.load_state_dict(state['model'])
epoch = state['epoch']
print(model.numel())
module.log(f'Current epoch is {epoch}', out=True)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
out = decode(model.generate(context, max_new_tokens=500)[0].tolist())

print(out)
module.log(out, out=True)
module.end_log()
