import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import Data
import numpy as np
from model import Wav2Vec2
import json
import os

seed = 1
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load dataset
dataset = []
label = []
if not os.path.exists("data.json"):
    data_path = "./Data_CNN_ravdess_savee/Data_CNN_ravdess_savee"
    class_path = os.listdir(data_path)
    for ind, i in enumerate(class_path):
        temp_path = []
        temp_label = []
        for j in os.listdir(os.path.join(data_path,i)):
            temp_path.append(os.path.join(data_path,i,j))
            label.append(ind)
        dataset.extend(temp_path)
        label.extend(temp_label)
    with open('data.json','w') as f:
        json.dump({'data':dataset,
                'label':label},f)   
else:
    with open('data.json','r') as f:
        temp = json.load(f) 
        dataset = temp['data']
        label = temp['label']

batch = 64
global data
data = Data(dataset, label)
data = DataLoader(data,batch_size=batch, shuffle=True)

# train model
def train():
    global data
    # model = Wav2Vec2(n_classes=8)
    model = torch.load('./model.pt')
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    lr = 0.00001
    optimizer = optim.SGD(model.parameters(), lr = lr)
    gamma = 0.9
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_list = list()
    acc_list = list()
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        print('the epoch is %d'%epoch)
        epoch_loss = 0
        acc = 0
        for _input, label in data:
            optimizer.zero_grad()
            _input = _input.float().cuda()
            label = label.squeeze().cuda()
            output = model(_input)
            loss = criterion(output, label.cuda())
            loss.backward()
            optimizer.step()
            acc += (output.cpu().detach().numpy().argmax(1) == label.cpu().detach().numpy()).sum()
            epoch_loss += loss.item()
        print('the epoch is %d the loss is %.3f'%(epoch,epoch_loss/len(data)))
        loss_list.append(epoch_loss/len(data))
        acc_list.append(acc/len(data)/batch)
        if epoch%1 == 0:
            scheduler.step()
            torch.save(model,'./model.pt')
    import json
    if os.path.exists("loss.json"):
        with open('loss.json','r') as f:
            data = json.load(f)
            loss_list = data['loss_list']
            acc_list = data['acc_list']
            loss_list.extend(loss_list)
            acc_list.extend(acc_list)
    with open("loss.json",'w') as f:
        json.dump({'loss_list':loss_list,
                    'acc_list':acc_list},f)    

def test(data_path,model_path = './model.pt'):
    dataset = []
    label = []
    batch = 2
    class_path = os.listdir(data_path)
    for ind, i in enumerate(class_path):
        temp_path = []
        temp_label = []
        for j in os.listdir(os.path.join(data_path,i)):
            temp_path.append(os.path.join(data_path,i,j))
            label.append(ind)
        dataset.extend(temp_path)
        label.extend(temp_label)
    data = Data(dataset, label)
    data = DataLoader(data,batch_size=batch, shuffle=True)    
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        acc = 0
        total = 0
        for _input, label in data:
            _input = _input.float().cuda()
            output = model(_input)
            label = label.squeeze().cuda()
            pre = np.array(output.cpu().detach().numpy()).argmax(1)
            acc += (output.cpu().detach().numpy().argmax(1) == label.cpu().detach().numpy()).sum()
            total += len(pre.reshape(-1))
    print('the acc is %.3f'%(acc/total))

# choose to train or  test
train()
# test("./Data_CNN_ravdess_savee/Data_CNN_ravdess_savee")

