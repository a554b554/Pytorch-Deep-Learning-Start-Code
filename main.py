import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataparser import parse_dataset
import sys

import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if val > self.max:
        	self.max = val
        self.avg = self.sum / self.count

def train(train_loader, model, epochs, batch_size, validation_loader=None):
		criterion = nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
		losses = AverageMeter()
		model.train()
		for ep in range(epochs):
			train_loader.init_iter(batch_size)
			n = train_loader.len()
			for i, (x, y) in enumerate(train_loader):
				input_var = torch.autograd.Variable(x)
				target_var = torch.autograd.Variable(y)

				output = model(input_var)
				loss = criterion(output, target_var)

				losses.update(loss.data[0], x.size(0))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				if i%1000 == 0:
					print('epoch:', ep,'/',epochs, 'iter:', i, '/', n, 'avg loss:', losses.avg, 'current loss:', loss.data[0])

			if validation_loader:
				val_loss = validation(validation_loader, model)
				print('validation loss:', val_loss.avg)

		return

def validation(validation_loader, model):
	criterion = nn.MSELoss()
	losses = AverageMeter()
	model.eval()
	validation_loader.init_iter(10, shuffle=False)
	for i, (x, y) in enumerate(validation_loader):
		input_var = torch.autograd.Variable(x)
		target_var = torch.autograd.Variable(y)
		output = model(input_var)
		loss = criterion(output, target_var)
		losses.update(loss.data[0], x.size(0))
		print('in:',input_var.data,'tar:',target_var.data,'out:',output.data*30000)
		break
	return losses

#define your data loader here
class DataLoader:
	def __init__(self):
		return

	def load(self, filename):
		self.x = torch.load(filename+'_x.pth')
		self.y = torch.load(filename+'_y.pth')

	def init_iter(self, batch_size, shuffle=True):
		self.batch_size = batch_size
		self.n = self.x.size(0)
		perm = torch.LongTensor(range(self.n))
		if shuffle:
			perm = torch.randperm(self.n)

		self.x = torch.index_select(self.x, 0, perm)
		self.y = torch.index_select(self.y, 0, perm)
	
		return

	def __iter__(self):
		self.cid = 0
		return self

	def __next__(self):
		if self.cid < self.n:
			cid = self.cid
			self.cid += self.batch_size
			return self.x[cid:cid+self.batch_size, :], self.y[cid:cid+self.batch_size]
		else:
			raise StopIteration()

	def len(self):
		return int(self.n / self.batch_size)

	def sample(self, batch_size):
		perm = torch.randperm(self.x.size(0))
		batch_index = perm[0:batch_size]
		x_out = torch.index_select(self.x, 0, batch_index)
		y_out = torch.index_select(self.y, 0, batch_index)
		return x_out, y_out

class dphiNet_bn(nn.Module):
	def __init__(self):
		super(dphiNet_bn, self).__init__()
		self.lin = nn.Sequential(
			nn.Linear(9, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(128, 6),
			)
		return

	def forward(self, x):
		x = self.lin(x)
		return x

class dphiNet(nn.Module):
	def __init__(self):
		super(dphiNet, self).__init__()
		self.lin = nn.Sequential(
			nn.Linear(9, 256),
			# nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 256),
			# nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 128),
			# nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(128, 6),
			)
		return

	def forward(self, x):
		x = self.lin(x)
		return x

class dphidrNet_bn(nn.Module):
	def __init__(self):
		super(dphidrNet_bn, self).__init__()
		self.lin = nn.Sequential(
			nn.Linear(5, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(128, 5),
			)
		return

	def forward(self, x):
		x = self.lin(x)
		return x

class Network:
	def __init__(self):
		self.network = dphidrNet_bn()
		self.network.load_state_dict(torch.load('modeldphidr_bn.pth'))
		self.network.eval()
		return

	def eval(self, input_array):
		input_tensor = torch.FloatTensor(input_array)
		input_tensor = input_tensor.view(-1, 5)
		input_var = torch.autograd.Variable(input_tensor)
		output_var = self.network(input_var)
		output = output_var.data
		output = output.view(-1)
		output*=30000
		return output.tolist()

def test_input():
	nn = Network()
	print(nn.eval([0, 0, 1, 0, 1]))
	return

def test_validation():
	validation_loader = DataLoader()
	validation_loader.load('dphidrval')


	model = dphidrNet_bn()
	model.load_state_dict(torch.load('modeldphidr_bn.pth'))

	validation(validation_loader, model)

if __name__ == '__main__':


	# test_input()
	test_validation()
	exit(0)

	train_loader = DataLoader()
	# train_loader.parsedata('../build_release/FOSSSim/woven_train/dphidr.txt')

	# torch.save(train_loader.x, 'dphidrtrain_x.pth')
	# torch.save(train_loader.y, 'dphidrtrain_y.pth')


	validation_loader = DataLoader()
	# validation_loader.parsedata('../build_release/FOSSSim/woven_train/dphidr_validate.txt')

	# print(torch.min(train_loader.x), torch.max(train_loader.x))
	# print(torch.min(train_loader.y), torch.max(train_loader.y))
	# print(torch.min(validation_loader.y), torch.max(validation_loader.y))

	# torch.save(validation_loader.x, 'dphidrval_x.pth')
	# torch.save(validation_loader.y, 'dphidrval_y.pth')
	# exit(0)

	train_loader.load('dphidrtrain')
	validation_loader.load('dphidrval')


	# print(torch.min(train_loader.x), torch.max(train_loader.x))
	# print(torch.min(train_loader.y), torch.max(train_loader.y))
	# print(torch.min(validation_loader.y), torch.max(validation_loader.y))

	model = dphidrNet_bn()
	train(train_loader=train_loader, model=model, epochs=30, batch_size=1024, validation_loader = validation_loader)

	# torch.save(model.state_dict(), 'modeldphidr_bn.pth')
	# model.load_state_dict(torch.load('modeldphidr_bn.pth'))
	# loss = validation(validation_loader, model)
	# print(loss.max, loss.avg)