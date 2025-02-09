
from libs.classes import GPT, GPTConfig
# from libs.data import DataClass
from libs.imp import Module as M
module = M(log_file='generate.out')

import torch
torch.manual_seed(1337)
import random


default_loc = "Practice/Addition/Models/model.pth.tar"

get_loc = input(f'Where is the current model (Default "{default_loc}")? ')
if get_loc == '': get_loc = default_loc

state = module.load_model(get_loc)
config: 'GPTConfig' = state['config']

model = GPT(config)
model.load_state_dict(state['model'])
module.log(f'Number of parameters is {model.numel()}', out=True)

epoch: int = state['epoch']
module.log(f'Current epoch is {epoch}', out=True)


# generate from the model
for i in range(1, 6):
	# nums = [random.randint(1, config.max_number-1) for _ in range(2)]
	nums = [int(i) for i in input().strip().split()]
	# print(nums)
	context = config.data.encode(nums, 0)[0].reshape(1, config.data.block_size)
	# print(context)
	out1 = config.data.decode(model.generate_mul(context, max_new_tokens=config.data.res_number)[0].tolist())
	out2 = config.data.decode(model.generate_max(context, max_new_tokens=config.data.res_number)[0].tolist())
	# print(eval(out1.strip('$').replace('=', '==', 1)))
	# print(eval(out2.strip('$').replace('=', '==', 1)))
	module.log(f'{i}: Multinomial- {out1}    Max- {out2}', out=True)


nums = [5207, 3699, 1000]
# print(nums)
context = config.data.encode(nums, 0)[0].reshape(1, config.data.block_size)
print(context)
out1 = config.data.decode(model.generate_mul(context, max_new_tokens=config.data.res_number)[0].tolist())
out2 = config.data.decode(model.generate_max(context, max_new_tokens=config.data.res_number)[0].tolist())
# print(eval(out1.strip('$').replace('=', '==', 1)))
# print(eval(out2.strip('$').replace('=', '==', 1)))
module.log(f'{i+1}: Multinomial- {out1}    Max- {out2}', out=True)

module.end_log()
