"""_summary_
A Py file which has several essential funcions
"""
import time

import torch

dont_save = False
log_file = 'Practice/GPT/Logs/log.out'

class Module:
	def __init__(self):
		self.start_time = time.time()
		self.curr_time = time.time()
		with open(log_file, 'w') as f:
			f.write('Program started at ' + str(self.start_time) + '\n')

	def time_since(self, start=True):
		if start:
			return time.time() - self.start_time
		return time.time() - self.curr_time
	
	def time(self):
		return time.time()

	def log(self, str: str, out: bool = False) -> None:
		if out:
			with open(log_file, 'a') as f:
				f.write(f'DEBUG (after {time.time() - self.curr_time} s): ' + str + '\n')
		print(f'DEBUG (after {time.time() - self.curr_time} s):', str)
		self.curr_time = time.time()

	def end_log(self) -> None:
		with open(log_file, 'a') as f:
			f.write('Program Ran Successfully' + '\n')
			f.write(f'took {time.time() - self.start_time} s to run' + '\n')
		print('Program Ran Successfully')
		print(f'took {time.time() - self.start_time} s to run')

	def save_model(self, state: dict, filename: str = 'model.pth.tar'):
		if dont_save: return
		self.log('Saving Model', out=True)
		# model: 'Module', optimizer: 'Optimizer', epoch: int
		# state = {
		# 	'model': model.state_dict(),
		# 	'optimizer': optimizer.state_dict(),
		# 	'epoch': epoch,
		# }
		torch.save(state, filename)

	def load_model(self, filename: str = 'model.pth.tar'):
		self.log('Loading Model', out=True)
		state = torch.load(filename)
		return state
		# model.load_state_dict(state['model'])
		# epoch = state['epoch']
		# if optimizer:
		# 	optimizer.load_state_dict(state['model'])
		# 	return model, optimizer, epoch
		
		# return model, epoch
