"""_summary_
A Py file which has several essential funcions
"""
import time
from time import strftime, localtime

import torch
import pandas as pd


dont_save = False

class Module:
	def __init__(self, log_file: str = 'log.out'):
		self.start_time = time.time()
		self.curr_time = time.time()
		self.log_file = 'Practice/Addition/Logs/' + log_file
		# with open(self.log_file, 'x') as f:
		# 	pass
		with open(self.log_file, 'w') as f:
			f.write('Program started at ' + str(self.time()) + '\n')

	def time_since(self, start=True):
		if start:
			return round(time.time() - self.start_time, 5)
		return round(time.time() - self.curr_time, 5)
	
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
	
	def save_tensor(self, t: torch.tensor, file: str):
		t_np = t.numpy() #convert to Numpy array
		df = pd.DataFrame(t_np) #convert to a dataframe
		df.to_csv(file,index=False) #save to file

		#Then, to reload:
	def load_tensor(self, file: str) -> torch.tensor:
		df = pd.read_csv(file)
		t = torch.from_numpy(df.values)
		return t


	def save_model(self, state: dict, filename: str = 'model.pth.tar'):
		if dont_save: return
		self.log('Saving Model', out=True)
		torch.save(state, filename)

	def load_model(self, filename: str = 'model.pth.tar'):
		self.log('Loading Model', out=True)
		state = torch.load(filename)
		return state
