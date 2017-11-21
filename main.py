import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
import csv
import copy
import numpy as np

import matplotlib.pyplot as plt
from scipy import interpolate


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, iters, batch_size):
		criterion = nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
		losses = AverageMeter()

		model.train()
		for i in range(iters):
			fs, ds = train_loader.sample(batch_size)
			input_var = torch.autograd.Variable(ds)
			target_var = torch.autograd.Variable(fs)

			output = model(input_var)
			loss = criterion(output, target_var)

			losses.update(loss.data[0], fs.size(0))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % 1000 == 0:
				print('iter', i, '/', iters)
				print('avg loss', losses.avg)




# def validate(val_loader, model, criterion):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input, volatile=True)
#         target_var = torch.autograd.Variable(target, volatile=True)

#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)

#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    i, len(val_loader), batch_time=batch_time, loss=losses,
#                    top1=top1, top5=top5))

#     print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
#           .format(top1=top1, top5=top5))

#     return top1.avg

class data_loader:
	def __init__(self, fs, ds):
		pass

	def sample(self, batch_size):
		pass
		return

class ConvNet3D(nn.Module):
	def __init__(self):
		super(ConvNet3D, self).__init__()
		self.features = nn.Sequential(
			nn.Conv3d(9, 32, 3),
			nn.ReLU(True), 
			nn.Conv3d(32, 16, 3),
			nn.ReLU(True))

		self.lin = nn.Sequential(
			nn.Linear(16*5*4*4, 256),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 1))

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 16*5*4*4)
		x = self.lin(x)
		return x


if __name__ == '__main__':
	# parsecsv('../data/voltage.csv')
	testinterp()
