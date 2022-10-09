import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import TCNN
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from MyDataset import MyDataset
from CTCDecoder import CTCDecoder

#################################### 参数配置 #######################################################################
parser = argparse.ArgumentParser(description='train parmeraters')
parser.add_argument('-m', '--MODELPATH', type=str, default='', help='预训练模型路径')
parser.add_argument('-d', '--DATASET', default='/mnt/S32/ssd/industry/barcode/MASSDATA/cutImgs', type=str)
parser.add_argument('-np', '--numprocesses', help='CTCBeamSearch使用的线程数量', default=8, type=int)
parser.add_argument('-re', '--regex', help='dataset file match regex', default='^\d{2}(-\d{2,})+_\d{4,}_(211412|211214|211232){1}(-[1-4]{6}){1,}-2331112.png$', type=str)
parser.add_argument('-e', '--epochs', help='training epochs', default=10, type=int)
parser.add_argument('-sr', '--sample_rate', help='dataset sample rate', default=0.05, type=float)
parser.add_argument('-lr', '--lr',help='learning rate',default=1e-3, type=float)
parser.add_argument('-b', '--batch_size', help='batch_size', default=24, type=int)
parser.add_argument('-bms', '--beamsize', help='beam_size', default=100, type=int)
parser.add_argument('-ws', '--workspace', help='result save path', default='./workspace', type=str)
parser.add_argument('-g', '--GPU', help='use GPU index', default='-1', type=str)
conf = parser.parse_args()
print(conf)

os.environ["CUDA_VISIBLE_DEVICES"] = conf.GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CPU")


####################################  数据集  #######################################################################
dataset = MyDataset(conf.DATASET, re_str=conf.regex, sample=conf.sample_rate)
print('数据集大小:', len(dataset))
dataload = DataLoader(dataset, conf.batch_size)
length = len(dataset)
train_size = int(length * 0.8)
valid_size = length - train_size
print("训练集大小:", train_size, "\t验证集大小:", length - train_size)
train_set, valid_set = random_split(dataset=dataset, lengths=[train_size, valid_size])

train_dataload = DataLoader(train_set, conf.batch_size, shuffle=True)
valid_dataload = DataLoader(valid_set, conf.batch_size)

#################################### 模型构建 #######################################################################
print('建立模型中...')
if torch.cuda.is_available():
    model = TCNN().cuda()
else:
    model = TCNN()
if conf.MODELPATH != '':
    print('加载预训练模型：', conf.MODELPATH)
    model.load_state_dict(torch.load(conf.MODELPATH))
print('模型建立成功')

optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 - (1 - 0.01) * e / (conf.epochs - 1))
ctcloss = nn.CTCLoss()
Decoder = CTCDecoder(num_processes=conf.numprocesses, beam_size=conf.beamsize)


#################################### 模型训练 #######################################################################
"""
Lr:统计训练集学习率
Loss:统计训练集的ctcloss
Acc_Greedy:统计验证集的准确率
Acc_BeamSearch:统计验证集的准确率
Acc_BeamSearch_Code128:统计验证集的准确率
Err_Greedy:统计误识率
Err_BeamSearch:统计误识率
Err_BeamSearch_Code128:统计误识率
"""
Loss = []
Lr = []
Acc_Greedy = []
Acc_BeamSearch = []
Acc_BeamSearch_Code128 = []
Err_Greedy = []
Err_BeamSearch = []
Err_BeamSearch_Code128 = []

for epoch in range(conf.epochs):
    model.train()
    pbar = tqdm(iter(train_dataload))
    pbar.set_description("Train {0}/{1}".format(epoch + 1, conf.epochs))
    for images, labels, lengths in pbar:
        # 模型及损失计算
        Lr.append(optimizer.param_groups[0]['lr'])
        if torch.cuda.is_available():
            output = model(images.cuda())
            ctc_loss = ctcloss(output, labels.cuda(), [450] * len(images), list(lengths))
        else:
            output = model(images)
            ctc_loss = ctcloss(output, labels, [450] * len(images), list(lengths))
        loss = ctc_loss
        Loss.append(loss.item())
        # 梯度下降
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({'Loss':'{:.4g}'.format(np.mean(Loss))})
    scheduler.step()
    # 验证集验证
    acc_greedy = 0
    acc_beamsearch = 0
    acc_beamsearch_code128 = 0
    err_greedy = 0
    err_beamsearch = 0
    err_beamsearch_code128 = 0
    total = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(iter(valid_dataload))
        pbar.set_description("Valid {0}/{1}".format(epoch + 1, conf.epochs))
        for images, labels, lengths in pbar:
            if torch.cuda.is_available():
                output = model(images.cuda())
            else:
                output = model(images)
            preds = output.permute(1, 0, 2)
            total += len(labels)
            # greedy
            codes, values, scores = Decoder.Greedy(preds)
            for code, label in zip(codes, labels):
                if len(code) == 0:
                    continue
                if code.equal(label.int()):
                    acc_greedy += 1
                else:
                    err_greedy += 1
            # beamsearch
            codes, values, scores = Decoder.BeamSearch(preds)
            for code, label in zip(codes, labels):
                if len(code) == 0:
                    continue
                if code.equal(label.int()):
                    acc_beamsearch += 1
                else:
                    err_beamsearch += 1
            # beamsearch_code128
            codes, values, scores = Decoder.BeamSearchCode128(preds)
            for code, label in zip(codes, labels):
                if len(code) == 0:
                    continue
                if code.equal(label.int()):
                    acc_beamsearch_code128 += 1
                else:
                    err_beamsearch_code128 += 1
            pbar.set_postfix({
                'Acc Greedy':'{:.4g}'.format(acc_greedy/total),
                'Acc BeamSearch':'{:.4g}'.format(acc_beamsearch/total),
                'Acc BeamSearchCode128':'{:.4g}'.format(acc_beamsearch_code128/total),
                'Err Greedy':'{:.4g}'.format(err_greedy/total),
                'Err BeamSearch':'{:.4g}'.format(err_beamsearch/total),
                'Err BeamSearch Code128':'{:.4g}'.format(err_beamsearch_code128/total),
            })
        Acc_Greedy.append(acc_greedy*100/total)
        Acc_BeamSearch.append(acc_beamsearch*100/total)
        Acc_BeamSearch_Code128.append(acc_beamsearch_code128*100/total)
        Err_Greedy.append(err_greedy*100/total)
        Err_BeamSearch.append(err_beamsearch*100/total)
        Err_BeamSearch_Code128.append(err_beamsearch_code128*100/total)
        model_file = conf.workspace + '/model_{0:}_{1:}.pt'.format(epoch, acc_beamsearch_code128)
        torch.save(model.state_dict(), model_file)

def savelist(filename, datalist):
    with open(filename, 'w') as f:
        for data in datalist:
            f.write(str(data))

savelist(conf.workspace + '/Loss.txt', Loss)
savelist(conf.workspace + '/Lr.txt', Lr)
savelist(conf.workspace + '/Acc_Greedy.txt', Acc_Greedy)
savelist(conf.workspace + '/Acc_BeamSearch.txt', Acc_BeamSearch)
savelist(conf.workspace + '/Acc_BeamSearch_Code128.txt', Acc_BeamSearch_Code128)
savelist(conf.workspace + '/Err_Greedy.txt', Err_Greedy)
savelist(conf.workspace + '/Err_BeamSearch.txt', Err_BeamSearch)
savelist(conf.workspace + '/Err_BeamSearch_Code128.txt', Err_BeamSearch_Code128)
print("Successfully!!!")
