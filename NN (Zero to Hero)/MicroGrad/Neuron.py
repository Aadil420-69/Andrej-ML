# pylint: disable=bad-indentation

import random
from Value import Value

class Neuron:
	def __init__(self, nin):
		self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
		self.b = Value(random.uniform(-1, 1))

	def __call__(self, x):
		act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
		out = act.tanh()
		return out

	def parameters(self):
		return self.w + [self.b]

class Layer:
	def __init__(self, nin, nout):
		self.neurons = [Neuron(nin) for _ in range(nout)]

	def __call__(self, x):
		outs = [n(x) for n in self.neurons]
		return outs[0] if len(outs) == 1 else outs

	def parameters(self):
		return [p for n in self.neurons for p in n.parameters()]

class MLP:
	def __init__(self, nin, nout):
		sz = [nin] + nout
		self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nout))]

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]


def tweak(data, iterations=20, inc=0.05):
	xs, ys, n = data
	for k in range(iterations):
		# Forward Pass
		ypred = [n(x) for x in xs]
		loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

		# Zero grad
		for p in n.parameters():
			p.grad = 0.0

		# Backward
		loss.backward()

		# Update
		for p in n.parameters():
			p.data -= inc * p.grad

		print(k, loss.data)
	return [xs, ys, n]


def main():
	n = MLP(3, [4, 4, 1])

	# Data
	xs = [
		[2.0, 3.0, -1.0],
		[3.0, -1.0, 0.5],
		[0.5, 1.0, 1.0],
		[1.0, 1.0, -1.0]
	]
	ys = [1.0, -1.0, -1.0, 1.0]

	print('Before Tweaking')
	ypred = [n(x) for x in xs]		# The prediction of our model
	loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))		# How wrong our model is
	print(loss)

	data = [xs, ys, n]
	data = tweak(data)

	print('After Tweaking')
	n = data[2]

	ypred = [n(x) for x in xs]		# The prediction of our model
	loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))		# How wrong our model is
	print(loss)
	print(ypred)


if __name__ == '__main__':
	main()
