import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TCNN
from tqdm import tqdm
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from CTCDecoder import CTCDecoder
MODELPATH = 'model_NoThetaLoss_cpu.pt'
dataset = MyDataset('./test_images', re_str='^\d{2}(-\d{2,})+_\d{4,}_(211412|211214|211232){1}(-[1-4]{6}){1,}-2331112.png$', sample=1/10)
print('数据集大小:', len(dataset))
dataload = DataLoader(dataset, 2)
model = TCNN()
# 定义网络
print('建立模型中...')
if torch.cuda.is_available():
    model = TCNN().cuda()
else:
    model = TCNN()
if MODELPATH != '':
    print('加载参数模型：', MODELPATH)
    model.load_state_dict(torch.load(MODELPATH))
print('模型加载成功')
Decoder = CTCDecoder(num_processes=1)
model.eval()
pbar = tqdm(iter(dataload))
for image, label, length in pbar:
    output = model(image)
    preds = output.permute(1, 0, 2)
    # greedy
    code, value, score = Decoder.Greedy(preds)
    print(code, value, score)
    code, value, score = Decoder.BeamSearch(preds)
    print(code, value, score)
    code, value, score = Decoder.BeamSearchCode128(preds)
    print(code, value, score)
