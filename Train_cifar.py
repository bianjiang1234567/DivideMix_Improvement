from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')#'asym_two_unbalanced_classes' #asym_two_unbalanced_classes_origin
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=1)
#parser.add_argument('--seed', default=123)#
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--opt', type=str, default='adam', help='optimize method')
parser.add_argument('--equal', type=bool, default=False, help='use equal')#啥也不输入是False 输入别的都转化成True
parser.add_argument('--net', type=str, default='CNN_small', help='network')
#parser.add_argument('--reweight_epoch', default=160, type=int)
parser.add_argument('--reweight_epoch', default=20, type=int)
parser.add_argument('--LDAM_DRW', default=False, type=bool)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)
            #Para_x, outputs_x = LDAM_DRE(outputs_x, targets_x, weight)
            pu = (torch.softmax(1 * outputs_u11, dim=1) + torch.softmax(1 * outputs_u12, dim=1) + torch.softmax(1 * outputs_u21, dim=1) + torch.softmax(1 * outputs_u22, dim=1)) / 4 #所有概率上求平均
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(1 * outputs_x, dim=1) + torch.softmax(1 * outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px # one hot
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           如果不是这一类的话会被弱化
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)  #训练数据的不同也保证了网络的divergence
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)# batch_size*2=128
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        if args.noise_mode == 'asym_two_unbalanced_classes':#网络会被不平衡的分布严重影响 预测概率会由【0.9 0.1】趋近【1 0】
            #pred_mean = torch.softmax(logits, dim=1).mean(0)  # 保留dim=1的维度的 输出概率更像是分布估计
            #print("pred_mean=",pred_mean)
            #prior = torch.ones(args.num_class) / args.num_class
            #prior = torch.tensor([0.9, 0.1])
            #prior = prior.cuda()
            #penalty = torch.sum(prior * torch.log(prior / pred_mean))  # 逐个元素相乘积
            penalty = 0
        #elif args.noise_mode == 'asym_two_unbalanced_classes_origin':
        else:
            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)#保留dim=1的维度的 输出概率更像是分布估计
            penalty = torch.sum(prior*torch.log(prior/pred_mean))#逐个元素相乘积

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)
        if args.noise_mode=='asym_two_unbalanced_classes':  # penalize confident prediction for asymmetric noise
            if args.LDAM_DRW:
                loss, _ = LDAM_DRE(outputs, labels)
                penalty = conf_penalty(outputs, labels)
                L = loss + penalty
            else:
                loss = CEloss(outputs, labels)
                L = loss
        elif args.noise_mode=='sym':
            loss = CEloss(outputs, labels)
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def eval_train(model,all_loss,epoch):
    model.eval()
    losses = torch.zeros(50000)
    Prob_Output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            #训练的时候减去一个数字，测试的时候直接输出这个数字
            '''if args.noise_mode == 'asym_two_unbalanced_classes':
                loss, outputs = LDAM_DRE_warmup(outputs,targets)
                Prob_Output_temp = nn.functional.softmax(outputs, dim=1)
                Prob_Output.append(Prob_Output_temp)
            else:
                Prob_Output_temp = nn.functional.softmax(outputs, dim=1)
                Prob_Output.append(Prob_Output_temp)
                loss = CE(outputs, targets)  
            '''
            Prob_Output_temp = nn.functional.softmax(1 * outputs, dim=1)#预测的时候不需要正则
            Prob_Output.append(Prob_Output_temp)

            if args.noise_mode == 'asym_two_unbalanced_classes':
                if args.LDAM_DRW:
                    weight = get_per_cls_wei(epoch)
                    loss, _ = LDAM_DRE(outputs, targets, weight, reduction='none')#warm up 与coguss cofinetune 后的 度量loss small loss trick
                else:
                    loss = CE(outputs, targets)
            else:
                loss = CE(outputs, targets)

            #loss = CE(outputs, targets)

            for b in range(inputs.size(0)):
                #print("b=", b)
                #print("index[b]=", index[b])
                losses[index[b]]=loss[b]#index[b]可以将每一个loss对应到每一个样本
    #Prob_Output_tensor = Prob_Output[0]
    #for i in range(1, len(Prob_Output)):
    Prob_Output_tensor = torch.cat(tuple(Prob_Output), dim=0)

    losses = (losses-losses.min())/(losses.max()-losses.min())
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    #print("prob1=", prob)
    #print("gmm.means_.argmin()=", gmm.means_.argmin())
    prob = prob[:,gmm.means_.argmin()] #属于小loss的概率是多少 获得的是真实无噪声样本的概率是多少 该样本无噪声的概率是多少
    #index_temp = np.argmax(prob, axis=1)
    #print("index_temp=", index_temp)
    #for i in range(len(prob)):
    #    prob[i] = prob[i, index_temp[i]]
    #prob = np.reshape(prob[:,0],newshape=(prob.shape[0]))
    #prob =  np.take(prob, index_temp, axis=1)
    #print("prob2=", prob)
    return prob,all_loss,Prob_Output_tensor

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):#targets_x softmax输出 detach
        if args.noise_mode == 'asym_two_unbalanced_classes':
            if args.LDAM_DRW:
                weight = get_per_cls_wei(epoch)
                _, outputs_x = LDAM_DRE(outputs_x, targets_x, weight)#output经过正则再与target(软标签)比较
                #Lx = -torch.mean(torch.sum(F.log_softmax(Para_x[0] * outputs_x, dim=1) * targets_x * Para_x[1], dim=1))
                Lx = -torch.mean(torch.sum(F.log_softmax(Para_s * outputs_x, dim=1) * targets_x * weight, dim=1))

                Para_u, outputs_u = LDAM_DRE(outputs_u, targets_u, weight)
                probs_u = torch.softmax(Para_s * outputs_u, dim=1)#预测相关
                Lu = torch.mean((probs_u - targets_u) ** 2 * weight)
            else:
                probs_u = torch.softmax(outputs_u, dim=1)

                Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
                Lu = torch.mean((probs_u - targets_u) ** 2)
        else:
            probs_u = torch.softmax(outputs_u, dim=1)

            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
            Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs, labels):
        if args.noise_mode == 'asym_two_unbalanced_classes':
            #weight = get_per_cls_wei(epoch)
            if args.LDAM_DRW:
                _, outputs = LDAM_DRE(outputs, labels)

        probs = torch.softmax(1 * outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
'''
def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model
'''
def create_model():
    if args.net == 'CNN_small':
        model = CNN_small(num_classes=args.num_class)#--lr 0.001
        model = model.cuda()
    else:
        model = ResNet18(num_classes=args.num_class)#--lr 0.02默认
        model = model.cuda()
    return model

class CNN_small(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#print("args.equal=",args.equal)
#print("args.equal.type=",type(args.equal))

stats_log=open('./checkpoint/%s_average_noise_%.1f_%s_equal_%s_%s_%s_LDAM_%s'%(args.dataset,args.r,args.noise_mode,args.equal ,args.net,args.opt, args.LDAM_DRW)+'_stats.txt','w')
test_log=open('./checkpoint/%s_average_noise_%.1f_%s_equal_%s_%s_%s_LDAM_%s'%(args.dataset,args.r,args.noise_mode,args.equal ,args.net,args.opt, args.LDAM_DRW)+'_acc.txt','w')

if args.dataset=='cifar10':
    warm_up = 10
    #warm_up = 20
    #warm_up = 30
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [args.lr] * args.num_epochs
beta1_plan = [mom1] * args.num_epochs
for i in range(args.epoch_decay_start, args.num_epochs):
    alpha_plan[i] = float(args.num_epochs - i) / (args.num_epochs - args.epoch_decay_start) * args.lr
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999)

if args.opt == 'adam':
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=args.lr)
else:
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
#if args.noise_mode=='asym_two_unbalanced_classes':
conf_penalty = NegEntropy()#防止塌缩为一类

#LDAMloss DRW
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, ):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))  # 归一化
        m_list = torch.FloatTensor(m_list).cuda()
        #m_list = torch.Tensor(m_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight.cuda()

    def forward(self, x, target, weight=None, reduction='mean'):
        index = torch.zeros_like(x, dtype=torch.uint8)  # batchsize classes
        if weight is not None:
            self.weight = weight
        #index = torch.zeros_like(x)  # batchsize classes
        #print("target.dtype=",target.dtype)
        if target.dtype != torch.int64:
            #print("!= torch.int64")
            _, ind = torch.max(target, dim=-1)
            labels = torch.zeros(size=(target.size()[0],1), dtype=torch.int64).cuda()#.type(torch.cuda.LongTensor)
            ind.unsqueeze_(dim=-1)
            #print("labels.size()=",labels.size())
            #print("ind.size()=", ind.size())
            #labels.scatter_(1, ind, 1)#第一个参数是self 最后一个1被广播了  index作为下标填到output对应的维度里面  index的下标填到src的维度里面
            index.scatter_(1, ind, 1)  # 在index的第一个维度  target是位置 onehot

            index_float = index.type(torch.cuda.FloatTensor)  # one hot
            # index_float = index.cuda()
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # None batchsize
            batch_m = batch_m.view((-1, 1))  # 广播
            x_m = x - batch_m

            output = torch.where(index, x_m, x)
            #print("output.size()=", output.size())
            return [self.s, self.weight], output

        else:
            #print("== torch.int64")
            index.scatter_(1, target.data.view(-1, 1), 1)  # 在index的第一个维度  target是位置 onehot

            index_float = index.type(torch.cuda.FloatTensor)  # one hot
            #index_float = index.cuda()
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # None batchsize
            batch_m = batch_m.view((-1, 1))  # 广播
            x_m = x - batch_m

            output = torch.where(index, x_m, x)
            return F.cross_entropy(self.s * output, target, weight=self.weight, reduction=reduction), output

#idx = args.num_epochs // 160#160个epoch之后使用Reweight
#betas = [0, 0.9999]
betas = [0] * args.num_epochs
for i in range(args.reweight_epoch, args.num_epochs):
    betas[i] = 0.9999
cls_num_list = [45000 , 5000]

def get_per_cls_wei(epoch):
    effective_num = 1.0 - np.power(betas[int(epoch)], cls_num_list)#等比数列求和的倒数  每个权重都是等比数列求和 数量是多少项求和  数量越多权重影响力越弱
    per_cls_weights = (1.0 - betas[int(epoch)]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return per_cls_weights

Para_s = 30#最佳的正则约束也要乘以这个参数
max_m = 0.5
LDAM_DRE = LDAMLoss(cls_num_list=cls_num_list, max_m=max_m, s=Para_s, weight=torch.FloatTensor([1.0]*len(cls_num_list)))

#cls_num_list_semiLoss = [45000, 5000]
#effective_num_semiLoss = 1.0 - np.power(betas[idx], cls_num_list_semiLoss)#等比数列求和每个权重都是等比数列求和 数量是多少项求和  数量越多权重影响力在衰减
#per_cls_weights_semiLoss = (1.0 - betas[idx]) / np.array(effective_num_semiLoss)
#per_cls_weights_semiLoss = per_cls_weights_semiLoss   / np.sum(per_cls_weights_semiLoss) * len(cls_num_list_semiLoss)
#per_cls_weights_semiLoss  = torch.FloatTensor(per_cls_weights_semiLoss).cuda(args.gpuid)
#LDAM_DRE_semiLoss = LDAMLoss(cls_num_list=cls_num_list_semiLoss, max_m=0.5, s=30, weight=1).cuda(args.gpuid)


all_loss = [[],[]] # save the history of losses from two networks

#for epoch in range(args.num_epochs+1):
for epoch in range(args.num_epochs + 1):
    if args.opt == 'adam' and not args.LDAM_DRW:
        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
    elif args.opt == 'adam' and args.noise_mode == 'asym_two_unbalanced_classes' and args.LDAM_DRW:
        lr = args.lr
        if epoch >= 80:
            lr /= 10
        elif epoch >= 160:
            lr /= 100
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
    else:
        lr=args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr



    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0],Prob_Output1=eval_train(net1,all_loss[0],epoch)
        prob2,all_loss[1],Prob_Output2=eval_train(net2,all_loss[1],epoch)
        #prob1, all_loss[0] = eval_train(net1, all_loss[0])
        #prob2, all_loss[1] = eval_train(net2, all_loss[1])
        predict_index1 = None
        predict_index2 = None
        if args.equal:
            _, predict_index1 = torch.max(Prob_Output1, dim=1)
            _, predict_index2 = torch.max(Prob_Output2, dim=1)

            #print("Prob_Output1.cpu()=", Prob_Output1.cpu())
            #print("predict_index1.cpu()=", predict_index1.cpu())
            equal_array = np.equal(predict_index1.cpu(), predict_index2.cpu())
            #print("equal_array=", equal_array)
            #print("equal_array.shape=", equal_array.shape)
            #print("prob1=", prob1)
            #print("prob1.shape=", prob1.shape)
            equal_array = equal_array.numpy()

            prob1_tmp = np.logical_and(prob1, equal_array)
            prob2_tmp = np.logical_and(prob2, equal_array)

            pred1 = (prob1_tmp > args.p_threshold)  # True or False  是否属于无噪音 真实样本 二维数组[[]]
            pred2 = (prob2_tmp > args.p_threshold)

        else:
            #p_threshold  要逐渐提高
            pred1 = (prob1 > args.p_threshold)  #True or False  是否属于无噪音 真实样本 二维数组[[]]
            pred2 = (prob2 > args.p_threshold)
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    test(epoch,net1,net2)  


