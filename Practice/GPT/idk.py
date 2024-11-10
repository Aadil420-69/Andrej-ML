from imp import Module as M
module = M()

import torch
from classes import GPT, GPTConfig, DataClass

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
vocab_size = 27

data = DataClass(torch.rand((2, 3)))
config = GPTConfig(data, vocab_size, batch_size, block_size,
				   max_iters, eval_intervals, learning_rate,
				   eval_intervals, n_embd, n_head,
				   n_layer, dropout, False)
model = GPT(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

state = module.load_model('model.pth.tar')
model.load_state_dict(state['model'])
optimizer.load_state_dict(state['model'])
epoch = state['epoch']
config = state['config']
