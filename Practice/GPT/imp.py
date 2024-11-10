"""_summary_
A Py file which has several essential funcions
"""
import time
from time import strftime, localtime

import torch

dont_save = False

class Module:
	def __init__(self, log_file: str = 'log.out'):
		self.start_time = time.time()
		self.curr_time = time.time()
		self.log_file = 'Practice/GPT/Logs/' + log_file
		with open(self.log_file, 'w') as f:
			f.write('Program started at ' + str(self.time()) + '\n')

	def time_since(self, start=True):
		if start:
			return time.time() - self.start_time
		return time.time() - self.curr_time
	
	def time(self):
		return strftime('%Y-%m-%d %H:%M:%S', localtime(time.time()))

	def log(self, str: str, out: bool = False) -> None:
		if out:
			with open(self.log_file, 'a') as f:
				f.write(self.time() + '\n')
				f.write(f'DEBUG (after {self.time_since()} s): ' + str + '\n')
		print(f'DEBUG (after {self.time_since(start=False)} s):', str)
		self.curr_time = time.time()

	def end_log(self) -> None:
		with open(self.log_file, 'a') as f:
			f.write(self.time() + '\n')
			f.write('Program Ran Successfully' + '\n')
			f.write(f'took {self.time_since()} s to run' + '\n')
		print('Program Ran Successfully')
		print(f'took {self.time_since()} s to run')

	def save_model(self, state: dict, filename: str = 'model.pth.tar'):
		if dont_save: return
		self.log('Saving Model', out=True)
		torch.save(state, filename)

	def load_model(self, filename: str = 'model.pth.tar'):
		self.log('Loading Model', out=True)
		state = torch.load(filename)
		return state
