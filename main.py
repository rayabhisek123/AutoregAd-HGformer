from __future__ import print_function
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import copy
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from cosine_lr_schedueler import CosineLRScheduler
# from torchlight import DictAction
import resource
from torch import linalg as LA
from importlib_metadata import requires
from torch import einsum, positive
import math
sys.path.insert(0,"./Hyperformer")



def ema_update(source, target, decay=0.99, start_itr=20, itr=None):
    if itr and itr<start_itr:
        decay = 0.0
    with torch.no_grad():
        for key, value in source.module.state_dict().items():
            target.state_dict()[key].copy_(target.state_dict()[key] * decay + value * (1 - decay))

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))



###################---------Random_Seed---------###################
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = False

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = True
###################---------Random_Seed---------###################




###################---------Instant_Class_import---------###################
def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))
###################---------Instant_Class_import---------###################




###################---------String_to_Boolean---------###################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
###################---------String_to_Boolean---------###################




###################---------Calculate_Loss---------###################
def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]
###################---------Calculate_Loss---------###################




###################---------Argument_Parser---------###################
def get_parser():
    
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='GRA Transformer')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default="./work_dir11/runs")
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # gra
    parser.add_argument('--joint-label', type=list, default=[], help='tells which group each joint belongs to')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=2, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--ema', action="store_true", default=False, help='ema weight for eval')
    parser.add_argument('--lambda_1', type=float, default=1e-4)
    parser.add_argument('--lambda_2', type=float, default=1e-1)
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=16, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--acc-info',  default=None, help='info of best accuracy')
    parser.add_argument('--step', type=int, default=[110, 120], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=[0,1], nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--momentum', type=float, default=0.9, help='nesterov momentum')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser
###################---------Argument_Parser---------###################




#####################-------------Processor-------------#####################
class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
        self.model = self.model.cuda(self.output_device)
        # self.start_epoch=self.arg.start_epoch
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.start_epoch=self.arg.start_epoch

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        # self.lr = self.arg.base_lr
        # self.best_acc = 0
        # self.best_acc_epoch = 0
        # self.start_epoch=self.arg.start_epoch
        
        
        if(self.arg.acc_info):
            info=torch.load(self.arg.acc_info)
            self.best_acc=info['best_acc']
            self.best_acc_epoch=info['best_acc_epoch']
            print('best accuracy loaded and updated')
            

        self.best_acc_ema = 0
        self.best_acc_epoch_ema = 0

        # self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)
        if self.arg.ema:
            Model = import_class(self.arg.model)
            self.model_ema = Model(**self.arg.model_args).cuda(self.output_device)
            ema_update(self.model, self.model_ema, itr=0)

    #####################-------------Load_Data-------------#####################
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.train_feeder_args),
                                                                    batch_size=self.arg.batch_size, 
                                                                    shuffle=True,
                                                                    pin_memory=True,
                                                                    prefetch_factor=16,
                                                                    num_workers=self.arg.num_worker,
                                                                    drop_last=True,
                                                                    worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.test_feeder_args),
                                                               batch_size=self.arg.test_batch_size,
                                                               shuffle=False,
                                                               num_workers=self.arg.num_worker,
                                                               drop_last=False,
                                                               worker_init_fn=init_seed)
    #####################-------------Load_Data-------------#####################
    
    #####################-------------Load_Model-------------#####################
    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        #print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.rec_loss=nn.MSELoss().cuda(output_device)
        # mem_size = self.data_loader['train'].dataset.__len__()
        # label_all = self.data_loader['train'].dataset.label
        # self.graphContrast = InfoNCEGraph(in_channels=3*25*25, out_channels=256, class_num=self.arg.model_args["num_class"], \
        #     mem_size=mem_size, label_all=label_all, T=self.arg.temperature).cuda(output_device)
        self.joint_label=[0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1]
        self.he=torch.ones(5).float()
        if self.arg.weights:
            states=torch.load(self.arg.model_saved_name + '-' + 'last_ckpt' + '.pth')
            self.model.load_state_dict(states['state_dict'])
            print('model weights loaded')
    #####################-------------Load_Model-------------#####################
    
    #####################-------------Optimizer-------------#####################
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.arg.base_lr, momentum=self.arg.momentum, nesterov=self.arg.nesterov, weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'NAdam':
            self.optimizer = optim.NAdam(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.arg.base_lr, yweight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        if self.arg.weights:
            states=torch.load(self.arg.model_saved_name + '-' + 'last_ckpt' + '.pth')
            self.optimizer.load_state_dict(states['optimizer'])
            self.start_epoch=states['epoch']
            print('optimizer loaded and epoch loaded')

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
    #####################-------------Optimizer-------------#####################
    
    #####################-------------Save_Arguments-------------#####################
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)
    #####################-------------Save_Arguments-------------#####################
    
    #####################-------------Adjust_Learnig_Rate-------------#####################
    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' or self.arg.optimizer == 'NAdam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    #####################-------------Adjust_Learnig_Rate-------------#####################
    
    #####################-------------Time_Related-------------#####################
    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    #####################-----------Time_Related-----------#####################
    
    #####################-------------Train-------------#####################
    def train(self, epoch, save_model=False):
        self.model.train()
        if self.arg.ema:
            self.model_ema.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        loss_value2 = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        # mix_precision is slower for this model!!!
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # torch.autograd.set_detect_anomaly(True)

        soft_label_emma = 0
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                # print('the output label is ' ,label)
                # print('the shape of label is ',label.shape)
            timer['dataloader'] += self.split_time()

            class SoftTargetCrossEntropy(nn.Module):

                def __init__(self):
                    super(SoftTargetCrossEntropy, self).__init__()

                def forward(self, x, target):
                    # because p has already been passed through softmax !
                    # cross entropy is non-symetric!! the order matters!!!
                    loss = -torch.sum(target * torch.log(x+1e-20), dim=-1)
                    return loss.mean()


            with torch.cuda.amp.autocast(enabled=use_amp):
                # output, z = self.model(data, F.one_hot(label, num_classes=self.model.module.num_class))
                output,inp,recon_emb,recon_emb1,qe,jl,he = self.model(data, 
                                                        F.one_hot(label, num_classes=self.model.num_class),
                                                        tr=True,
                                                        jl=self.joint_label,
                                                        he=self.he.cuda(self.output_device))
                #print('model n fwd prop done')

                loss = self.loss(output, label)
                recon_emb=torch.flatten(recon_emb,start_dim=1,end_dim=-1)
                inp=torch.flatten(inp,start_dim=1,end_dim=-1)
                # loss2 = torch.zeros_like(loss).cuda(loss.device)
                recon_loss=self.rec_loss(recon_emb,inp)
                recon_emb1=torch.flatten(recon_emb1,start_dim=1,end_dim=-1)
                recon_loss1=self.rec_loss(recon_emb1,inp)
                #print('recon_f_loss is ', recon_f_loss.shape)
                #print('the values of loss is', recon_f_loss.item())
                #print('the value cel loss is',loss.item())
                
            
            #CalculateLoss
            # loss += loss2
            # loss3=recon_f_loss
            loss+=0.3*recon_loss
            loss+=0.3*recon_loss1
            loss+=0.3*qe
            #print('Loss = ',loss.item())
            self.joint_label = jl
            
            #BackwardProp
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            self.he = copy.copy(he)
            scaler.step(self.optimizer)
            scaler.update()
            

            loss_value.append(loss.data.item())
            loss_value2.append(loss2.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss2', loss2.data.item(), self.global_step)
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        self.print_log('\tMean training loss: {:.4f}. loss2: {:.4f}. Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(loss_value2), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # if save_model:
        #     # comb_state_dict
        #     state_dict = self.model.state_dict()
        #     weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        #     comb_state_dict={
        #     'train_loss':loss,
        #     'epoch': epoch + 1,
        #     'state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     }
            
        #     torch.save(comb_state_dict, self.arg.model_saved_name + '-' + 'last_ckpt' + '.pth')
    #####################--------------Train--------------#####################
    
    
    #####################-------------Evaluation-------------#####################
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None,save_model=False, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        if self.arg.ema:
            self.model_ema.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            loss_value_ema = []
            score_frag_ema = []
            label_list_ema = []
            pred_list_ema = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                if arg.ema:
                    label_list_ema.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    
                    # output, z = self.model(data, F.one_hot(label, num_classes=self.model.module.num_class))
                    output,_,_,z,_,_ = self.model(data, F.one_hot(label, num_classes=self.model.num_class),tr=False,jl=self.joint_label,he=self.he.cuda(self.output_device))
                    
                    if arg.ema:
                        self.model_ema.cuda(self.output_device)
                        # output_ema, z_ema = self.model_ema(data, F.one_hot(label, num_classes=self.model.module.num_class))
                        output_ema, z_ema = self.model_ema(data, F.one_hot(label, num_classes=self.model.num_class))

                    loss = self.loss(output, label)
                    if arg.ema:
                        loss_ema = self.loss(output_ema, label)

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    if arg.ema:
                        score_frag_ema.append(output_ema.data.cpu().numpy())
                        loss_value_ema.append(loss_ema.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

                    if arg.ema:
                        _, predict_label_ema = torch.max(output_ema.data, 1)
                        pred_list_ema.append(predict_label_ema.data.cpu().numpy())

                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            if self.arg.ema:
                score_ema = np.concatenate(score_frag_ema)
                loss_ema = np.mean(loss_value_ema)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            if self.arg.ema:
                accuracy_ema = self.data_loader[ln].dataset.top_k(score_ema, 1)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                
                weights_path=os.path.join(self.arg.work_dir, 'runs-'+'best_model.pth')
                torch.save(self.model.state_dict(),weights_path)
                print('best_acc updated and best model saved')
            if save_model:
                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                comb_state_dict={'train_loss':loss,'epoch': epoch + 1,'state_dict': self.model.state_dict(),'optimizer': self.optimizer.state_dict(),}
                torch.save(comb_state_dict, self.arg.model_saved_name + '-' + 'last_ckpt' + '.pth')
            # comb_state_dict
            # state_dict = self.model.state_dict()
            # weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            # comb_state_dict={
            # 'train_loss':loss,
            # 'epoch': epoch + 1,
            # 'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # }
            
            # torch.save(comb_state_dict, self.arg.model_saved_name + '-' + 'last_ckpt' + '.pth')
                        
                

            if self.arg.ema:
                if accuracy_ema > self.best_acc_ema:
                    self.best_acc_ema = accuracy_ema
                    self.best_acc_epoch_ema = epoch + 1
            acc_state_dict= {'best_acc':self.best_acc,
            'best_acc_epoch':self.best_acc_epoch
            }
            torch.save(acc_state_dict,self.arg.work_dir+'/best_acc_info.pth')
            print('best acc info saved')

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.ema:
                print('Accuracy_ema: ', accuracy, ' model_ema: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
                if arg.ema:
                    self.val_writer.add_scalar('loss_ema', loss_ema, self.global_step)
                    self.val_writer.add_scalar('acc_ema', accuracy_ema, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            
            if self.arg.ema:
                score_dict_ema = dict(zip(self.data_loader[ln].dataset.sample_name, score_ema))
                
            self.print_log('\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value)))
                
            if self.arg.ema:
                self.print_log('\tMean {} loss_ema of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value_ema)))
                    
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
                    
            if arg.ema:
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}_ema: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score_ema, k)))
                        
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
                    
            if arg.ema:
                if save_score:
                    with open('{}/epoch{}_{}_score_ema.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                        pickle.dump(score_dict_ema, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            #with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
            #    writer = csv.writer(f)
            #    writer.writerow(each_acc)
            #    writer.writerows(confusion)

            if arg.ema:
                # acc for each class:
                label_list_ema = np.concatenate(label_list_ema)
                pred_list_ema = np.concatenate(pred_list_ema)
                confusion_ema = confusion_matrix(label_list_ema, pred_list_ema)
                list_diag_ema = np.diag(confusion_ema)
                list_raw_sum_ema = np.sum(confusion_ema, axis=1)
                each_acc_ema = list_diag_ema / list_raw_sum_ema
                with open('{}/epoch{}_{}_each_class_acc_ema.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(each_acc_ema)
                    writer.writerows(confusion_ema)
    ###################-------------Evaluation-------------###################
    
    #####################-------------Start-------------#####################
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                # self.lr_scheduler.step(epoch)
                # self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                self.train(epoch, save_model=save_model)
                if self.arg.ema:
                    ema_update(self.model, self.model_ema, itr=epoch)
                self.eval(epoch, save_score=self.arg.save_score,save_model=save_model ,loader_name=['test'])

            # test the best model
            #weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights_path=os.path.join(self.arg.work_dir, 'runs-'+'best_model.pth')
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            # mask = self.model.module.joint_label
            #
            # A = torch.tensor(self.model.module.graph.A).cuda(mask.device).float()
            # A[A!=0] = 1
            #
            # ind = torch.argmax(mask, dim=0)
            # print(ind)


            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
    #####################-------------Start-------------#####################
    
#####################-------------Processor-------------#####################



#####################-------------Main()-------------#####################
if __name__ == '__main__':
    
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
#####################-------------Main()-------------#####################
