
print(eval('10+11==21'))

from libs.imp import Module as M
module = M()

f = open('Practice/Addition/exit2.txt', 'r')
while True:
	t = f.read()
	# print(t)
	if t:
		module.log('Exiting')
		break
		# with open('Practice/Addition/exit.txt', 'w') as f:

module.end_log()
