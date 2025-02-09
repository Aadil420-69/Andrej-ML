import torch


class DataClass:
	def __init__(self, nums: torch.Tensor, res_number: int,
			  block_size: int, split: int = 0.9):
		# Create mappings
		text = list('0123456789+=$')
		self.itos = dict(enumerate(text))
		self.stoi = {s: i for i, s in enumerate(text)}
		self.vocab_size = len(text)

		self.res_number = res_number
		self.block_size = block_size

		n = int(split * len(nums))
		self.train_data = nums[:n]
		self.val_data = nums[n:]


	# Create function to encode and decode
	def e(self, s: str) -> list[int]:
		return list(self.stoi[char] for char in s)

	def encode(self, arr: list[int] | torch.Tensor, r: int, tensor: bool = False) \
		-> tuple[list[int], list[int]] | torch.Tensor:

		if tensor: arr = arr.tolist()
		# print(f'{block_size=}\n{len(arr)=}\n{sum(len(str(num)) for num in arr)=}')
		n = self.block_size - len(arr) - sum(len(str(num)) for num in arr)
		if n < 0:
			print('ERROR')
			print(len(arr))
			print(type(arr))
			assert n > 0

		out = self.e('$')*n
		for char in arr:
			out += self.e(str(char) + '+')
		# out = out[:-1] + e('=')

		summ = str(sum(arr))
		out = out[:-1] + self.e('=' + '0'*(self.res_number - len(summ)) + summ)
		out1 = out[r:][: self.block_size]
		out2 = out[r+1:][: self.block_size]

		return torch.tensor((out1, out2), dtype=torch.long)
		# if tensor:
		# return (out1, out2)

	def decode(self, arr: list[int]) -> str:
		return ''.join(self.itos[char] for char in arr if self.itos[char] != '$')


	def decode_dollar(self, arr: list[int]) -> str:
		return ''.join(self.itos[char] for char in arr)

