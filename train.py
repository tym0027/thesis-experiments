import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import time
# from vgg import VGGi
from models import *
# from convnet import ConvNet
# from resnet import ResNet18, ResNet50

import numpy as np
import random
import time
from datetime import datetime

from models.IndependentComponent import * # IndependentComponentLayer

from utils import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', # originally 64
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', # orgiinally 256
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.3, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')

parser.add_argument('--distill', action='store_true', default=False ,
                    help='if distillation')
parser.add_argument('--teacher_path', default=False,
                    help='the path of teacher model')
parser.add_argument('--temperature', default=3, type=float,
                    help='temperature of distillation')
parser.add_argument('--kd_coefficient', default=0.5, type=float,
                    help='loss coefficient of knowledge distillation')
parser.add_argument('--teacharch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--teachdepth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')

parser.add_argument('--dataset', default="cifar10", type=str, help='select dataset...')
parser.add_argument('--save_name', default="test", type=str, help="input a name to refer to your experiment...")
parser.add_argument('--resume', action='store_true', default=False, help='if resuming training')
parser.add_argument('--resume_path', default="", type=str, help='path to resume from...')
parser.add_argument('--icl',  action='store_true', default=False, help='if you want to use an ICL...')
parser.add_argument('--icl_aug',  action='store_true', default=False, help='if you want to use an ICL as a data augmenter...')
parser.add_argument('--dynamic',  action='store_true', default=False, help='if you want to use a dynamic ICL...')
parser.add_argument('--icl_type', type=str, default='BN', help='if you want to use an ICL type other than BN...')

parser.add_argument('--cuda_num', type=int, default=0, help='Set which gpu to use...')
parser.add_argument('--test_only', action='store_true', default=False, help='Only test the model...')

args = parser.parse_args()
# if args.distill == "False":
#     args.distill = False
# else:
#     args.distill = True

args.cuda = not args.no_cuda and torch.cuda.is_available()


def log(s, timestamp=True):
    if not os.path.exists('logs/' + args.save_name):
        os.mkdir('logs/' + args.save_name)
    if timestamp:
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
        f = open('logs/' + args.save_name + '/log.txt', 'a')
        f.write(timestamp + ": " + s.replace("\n","\n" + timestamp + ":") + '\n')
    else:
        f = open('logs/' + args.save_name + '/log.txt', 'a')
        f.write(s + '\n')

    f.close()


### SET RANDOM SEEDs - MAKE REPEATABLE
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


transform_train = transforms.Compose([
    # transforms.Resize((32,32)),
    # transforms.RandomCrop(28, padding=0),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    # transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

'''
transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
'''

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

'''
transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
'''

if args.dataset == "mnist":
    # nlog('==> using MNIST data..', True)
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, drop_last=False, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, drop_last=False, shuffle=True, num_workers=1)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

elif args.dataset == "fmnist":
    # log('==> using Fashion MNIST data..', True)
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, drop_last=True, shuffle=True, num_workers=1)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, drop_last=True, shuffle=True, num_workers=1)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

elif args.dataset == "cifar10":
    # log('==> using CIFAR10 data..', True)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, drop_last=False, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, drop_last=False, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
elif args.dataset == "cifar100":
    # log('==> using CIFAR100 data..', True)
    trainloader = torch.utils.data.DataLoader(datasets.CIFAR100('./data', train=True, download=True, transform=transform_train),batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)

    testloader = torch.utils.data.DataLoader(datasets.CIFAR100('./data', train=False, transform=transform_test),batch_size=args.test_batch_size, drop_last=True, shuffle=True, **kwargs)

    classes = range(0, 100)

if args.cuda:
    if args.arch == "vgg" and not args.icl:
        if args.depth == 16:
            # model = VGG(depth=16, init_weights=True, cfg=None)
            model = VGG('VGG16', len(classes))
        elif args.depth == 19:
            # model = VGG(depth=19, init_weights=True, cfg=None)
            model = VGG('VGG19', num_classes=len(classes))
        else:
            sys.exit("vgg doesn't have those depth!")
    elif args.arch == "vgg" and args.icl:
        if args.depth == 16:
            # model = VGG(depth=16, init_weights=True, cfg=None)
            model = ICVGG('VGG16', len(classes))
        elif args.depth == 19:
            # model = VGG(depth=19, init_weights=True, cfg=None)
            model = ICVGG('VGG19', num_classes=len(classes))
        else:
            sys.exit("vgg doesn't have those depth!")
    elif args.arch == "lenet":
            # model = ICLeNet()
            model = LeNet()
    elif args.arch == "resnet" and not args.icl:
        if args.depth == 18:
            model = ResNet18()
        elif args.depth == 50:
            # model = ResNet50()
            model = ResNet50(len(classes))
        else:
            sys.exit("resnet doesn't implement those depth!")
    elif args.arch == "resnet" and args.icl:
        if args.depth == 18:
            model = ICResNet18()
        elif args.depth == 50:
            # model = ResNet50()
            model = ICResNet50(len(classes))
        else:
            sys.exit("resnet doesn't implement those depth!")


    # elif args.arch == "convnet":
    #     args.depth = 4
    #     model = ConvNet()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    model.cuda(args.cuda_num)

ic_net = ''

if args.icl_aug:
    ic_net = IndependentComponentLayer("STATIC-BATCH", 32, 32).cuda(args.cuda_num)

# print("P value is:", ic_net.get_p())
# ic_net.set_p(50,10)
# print("P value is:", ic_net.get_p())
# input()

if args.cuda and args.distill:
    #   crete teacher model.
    if args.teacharch == "vgg":
        if args.teachdepth == 16:
            teacher = VGG(depth=16, init_weights=True, cfg=None)
        elif args.teachdepth == 19:
            teacher = VGG(depth=19, init_weights=True, cfg=None)
        else:
            sys.exit("vgg doesn't have those depth!")
    elif args.teacharch == "resnet":
        if args.teachdepth == 18:
            teacher = ResNet18()
        elif args.teachdepth == 50:
            teacher = ResNet50()
        else:
            sys.exit("resnet doesn't implement those depth!")
    else:
        sys.exit("unknown network")
    # teacher = ResNet18()
    teacher.load_state_dict(torch.load(args.teacher_path))
    teacher.cuda(args.cuda_num)
    if args.multi_gpu:
        teacher = torch.nn.DataParallel(teacher)


#############

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda(args.cuda_num)
# criterion = nn.CrossEntropyLoss()


# args.smooth = args.smooth_eps > 0.0
# args.mixup = config.alpha > 0.0

optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

optimizer = None
if(args.optmzr == 'sgd'):
    optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr,momentum=0.9, weight_decay=1e-4)
    print("hello")
elif(args.optmzr =='adam'):
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
    print("hello")

print("goodbye")


scheduler = None
if args.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(trainloader), eta_min=4e-08)
elif args.lr_scheduler == 'default':
    # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
    epoch_milestones = [150, 250, 350]

    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*len(trainloader) for i in epoch_milestones], gamma=0.1)
else:
    raise Exception("unknown lr scheduler")

if args.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr/args.warmup_lr, total_iter=args.warmup_epochs*len(trainloader), after_scheduler=scheduler)


#############

def train(trainloader,criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        optimizer.zero_grad()
        if args.dataset == "mnist" or args.dataset == "fmnist":
            input = input.repeat(1, 3, 1, 1)
        # print()
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        scheduler.step()

        input = input.cuda(args.cuda_num, non_blocking=True)
        target = target.cuda(args.cuda_num, non_blocking=True)

        if args.icl_aug:
            ic_net.train()
            input = ic_net.forward(input)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        output = ''
        # compute output
        # output, _, _ = model.forward(input, 0.0, 0.0, 0.0)
        #print(input.shape)
        # print(input)
        # output, _, _ = model.forward(input)
        
        
        
        if args.arch == "lllenet":
            output, _, _ = model(input)
        elif args.arch == "vgg":
            output = model.forward(input)
        else:
            output = model.forward(input)
        # print(target.shape)
        # print(target)
        # print(output.shape)
        # print(output)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~")
        # output = model.forward(input)

        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
        else:
            # print("smooth")
            # print(args.smooth)
            ce_loss = criterion(output, target, smooth=args.smooth)
            # ce_loss = criterion(output, target)
        ce_loss.backward()

        # compute teacher output
        if args.distill:
            with torch.no_grad():
                teacher_output = teacher(input)
            distill_loss = KL(output, teacher_output)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.distill:
            losses.update(ce_loss.item() * (1 - args.kd_coefficient) + distill_loss.item() * args.kd_coefficient,
                          input.size(0))
        else:
            losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        if args.distill:
            overall_loss = ce_loss * (1 - args.kd_coefficient) + distill_loss * args.kd_coefficient
            overall_loss.backward()
        else:
            pass # rint("??????")
            # ce_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            s = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\tAcc@1 {top1.val:.3f} ({top1.avg:.3f}) Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(trainloader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5)

            log(s, True)
            '''
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            '''
            print(s)

        # UPDATE DYNAMIC P VALUES
        ''' not tetsing dynamics yet...
        if args.icl and args.dynamic and epoch > 0:
            # print("\n\nP value was:", ic_net.get_p())
            ic_net.update_p()
            s = "\tP value is: " + str(ic_net.get_p())
            print(s)
            log(s, True)
        '''


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        if args.cuda:
            data, target = data.cuda(args.cuda_num), target.cuda(args.cuda_num)
        
        if args.dataset == "mnist" or args.dataset == "fmnist":
            data = data.repeat(1, 3, 1, 1)
        
        data, target = Variable(data, volatile=True), Variable(target)

        if args.icl_aug:
            ic_net.eval()
            data = ic_net.forward(data)
        
        # print(data.shape)
        # print(target.shape)
        '''
        output = ''
        if args.arch == "llllenet":
            output, _, _ = model(data)
        elif args.arch == "vgg":
            output = model.forward(data)
        else:
            output = model.forward(data)

        # print(output.shape)
        '''

        output = model.forward(data)
        criterion = nn.CrossEntropyLoss()
        # test_loss = criterion(output, target)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # .data[0].item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    s = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        float(100. * correct) / float(len(testloader.dataset)))
    print(s)
    log(s, True)
    '''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        float(100. * correct) / float(len(testloader.dataset))))
    '''
    return (float(100 * correct) / float(len(testloader.dataset))), correct, float(len(testloader.dataset))


def load_checkpoint(path, model, ic_net, optimizer):
    loadedcheckpoint = torch.load(path)

    try:
        model.load_state_dict(loadedcheckpoint['state_dict'])
    except:
        print("no state dict!!!")

    try:
        start_epoch = loadedcheckpoint['epoch']
    except:
        print("no epoch!!!")

    try:
        optimizer.load_state_dict(loadedcheckpoint['optimizer'])
    except:
        print("no optimizer!!!")
    try:
        scheduler.load_state_dict(loadedcheckpoint['scheduler'])
    except:
        print("No scheduler!!!")
    try:
        ic_net.load_state_dict(loadedcheckpoint['icl'])
    except:
        print("No IC net!!!")
    return model, optimizer, scheduler, start_epoch


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def main():
    global optimizer, model, criterion, scheduler
    all_acc = [0.000]
    start_epoch = 0

    if args.resume:
        # Load checkpoint.
        print('resuming from checkpoint... ' + str(args.resume_path))
        log('resuming from checkpoint... ' + str(args.resume_path))
        assert os.path.isdir('logs/' + args.save_name), 'Error: no checkpoint directory found!'
        model, optimizer, scheduler, start_epoch = load_checkpoint(args.resume_path, model, ic_net, optimizer)
   
    if args.test_only:
        prec1, correct, total = test()
        # all_acc = [prec1]
        print("Final Accuracy: {acc} from [{correct}/{total}]".format(acc=prec1,correct=correct,total=total))
        exit()

    for epoch in range(start_epoch, args.epochs):
        # if epoch in [args.epochs * 0.26, args.epochs * 0.4, args.epochs * 0.6, args.epochs * 0.83]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        # if args.lr_scheduler == "default":
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * (0.5 ** (epoch // args.lr_decay))
        # elif args.lr_scheduler == "cosine":
        #     scheduler.step()

        if args.resume and (start_epoch - epoch == 0):
            prec1, correct, total = test()
            all_acc = [prec1]

            if args.icl and args.dynamic:
                ic_net.set_a_and_b(correct, total - correct)
        
        train(trainloader,criterion, optimizer, epoch)
        prec1, correct, total = test()

        if prec1 > max(all_acc):
            s = "\n>_ Got better accuracy at epoch [{a}], saving model with accuracy {b:.3f}% now...\n".format(a=str(epoch),b=prec1)
            log(s, True)
            print(s)
            # print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            # torch.save(model.state_dict(), "./logs/" + args.save_name + "/{}_{}{}_acc_{:.3f}_{}_{}_tem_{}_epoch{}.pt".format(args.dataset, args.arch, args.depth, prec1, args.optmzr, args.teacharch, args.temperature, epoch))
            if args.icl_aug:
                save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'icl'       : ic_net.state_dict(),
                        }, filename="./logs/" + args.save_name + "/" + args.arch+ "-" + str(int(prec1)) + "-" + str(epoch)+'.pth')
            else:
                save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        }, filename="./logs/" + args.save_name + "/" + args.arch+ "-" + str(int(prec1)) + "-" + str(epoch)+'.pth')

            
            s = "\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_acc))
            print(s)
            log(s, True)
            # print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_acc)))
            if len(all_acc) > 1:
                os.remove("./logs/" + args.save_name +  "/" + args.arch + "-" + str(int(prev_prec1)) + "-" + str(last_saved_epoch)+'.pth')
            last_saved_epoch = epoch
            prev_prec1 = prec1

            # UPDATE A & B FOR BETA DIST.
            if args.icl and args.dynamic:
                print("\nupdating a and b for beta distribution...")
                ic_net.set_a_and_b(correct, total - correct)

        # UPDATE DYNAMIC P VALUES
        if args.icl and args.dynamic:
            print("\n\nP value was:", ic_net.get_p())
            ic_net.update_p()
            s = "\nP value is: " + str(ic_net.get_p())
            print(s)
            log(s, True)



        all_acc.append(prec1)

        if prec1 >= 99.99:
            s = "accuracy is high ENOUGH!"
            print(s)
            log(s, True)
            # print("accuracy is high ENOUGH!")
            break

    s = "Best accuracy: " + str(max(all_acc))
    print(s)
    log(s, True)
    # print("Best accuracy: " + str(max(all_acc)))


if __name__ == '__main__':
    main()
