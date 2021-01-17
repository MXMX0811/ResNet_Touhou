import warnings
import csv
import torch
import time
import cv2
import os
import torch.utils.data as Data
import torchvision.transforms as transforms
import ResNet
from torch import nn
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
print(device)


def dataReader():
    Input = []
    label = []
    for i in range(1, 11):
        data_path = '../Dataset/Capture_' + str(i) + '/'
        label_path = '../Dataset/KeyCapture_' + str(i) + '.csv'
        for file in os.listdir(data_path):
            img = cv2.imread(data_path + file)
            # img = cv2.resize(img, (122, 141))
            transf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ]
            )
            img = transf(img)  # tensor数据格式是torch(C,H,W)
            Input.append(img)

        with open(label_path, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [list(map(int, row)) for row in reader]
        label.extend(rows)

    return Input, label


data_x, data_y = dataReader()
for i in range(len(data_y)):
    if data_y[i][0] != 0 or data_y[i][1] != 0:
        data_x.append(data_x[i])
        data_x.append(data_x[i])
        data_x.append(data_x[i])
        data_y.append(data_y[i])
        data_y.append(data_y[i])
        data_y.append(data_y[i])

    if data_y[i][2] != 0 or data_y[i][3] != 0:
        data_x.append(data_x[i])
        data_x.append(data_x[i])
        data_y.append(data_y[i])
        data_y.append(data_y[i])

data_x = torch.stack(data_x, dim=0)
data_y = torch.FloatTensor(data_y)
print(data_x.shape)
print(data_y.shape)


ResNet = ResNet.ResNet18()
ResNet.to(device)
epochs = 200
# 定义loss和optimizer
optimizer = optim.Adam(ResNet.parameters(), lr=0.0005, weight_decay=0.0)
criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, threshold=1e-3)

# 训练
batch_size = 64
torch_dataset = Data.TensorDataset(data_x, data_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,  # 批大小
    shuffle=True,
)

for epoch in range(1, epochs + 1):
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        output = ResNet(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TimeStr = time.asctime(time.localtime(time.time()))
    print('Epoch: {} --- {} --- '.format(epoch, TimeStr))
    print('Train Loss of the model: {}'.format(loss))
    print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    # 调整学习率
    scheduler.step(loss)

torch.save(ResNet.state_dict(), 'ResNet.pkl')

