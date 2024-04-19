import numpy as np
# from sklearn.metrics import recall_score,confusion_matrix,precision_score,accuracy_score,f1_score,roc_auc_score
import torch
import cv2
import time
import matplotlib
import math
from matplotlib import pyplot as plt

import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unetuc import Unet
# from dataset_CASIA import LiverDataset_train,LiverDataset_test
# from dataset_IITD import LiverDataset_train,LiverDataset_test
from dataset_UBIRIS import LiverDataset_train, LiverDataset_test
# from Unetplus  import NestNet
from loss_function import DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),  # -> [0,1]
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

x_transforms11 = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),  # -> [0,1]
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

y_transforms = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.Grayscale(1),
    transforms.ToTensor()
])
# 参数解析器,用来解析从终端读取的命令
parse = argparse.ArgumentParser()


def train_model(model, criterion, optimizer, dataload, num_epochs=100):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))

        with open("loss.txt", "a+") as f:
            f.write("epoch %d loss:%0.3f" % (epoch, epoch_loss) + "\n")

        with open("loss.txt", "a+") as f:
            f.write("epoch %d loss:%0.3f" % (epoch, epoch_loss))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'weights_%d.pth' % epoch, _use_new_zipfile_serialization=False)
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model


# 训练模型
def train():
    model = Unet(3, 1).to(device)
    # model=NestNet(3, 1).to(device)
    if torch.cuda.is_available():
        model = model.cuda()
    batch_size = 1
    criterion = torch.nn.BCELoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset_train(transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_model(model, criterion, optimizer, dataloaders)


# 显示模型的输出结果
def test():
    I = 0
    P_score = []
    R_scores = []
    ACC_score = []
    IOU_score = []
    F1_score = []
    NICE1_score = []
    NICE2_score = []

    model = Unet(3, 1).to(device)
    # model = NestNet(3, 1)
    # model = Dense_Unet(1).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = LiverDataset_test(transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    # plt.ion()
    a = time.time()
    with torch.no_grad():
        for x, true_mask in dataloaders:
            x1 = x
            x = x.to(device)
            # y=model(x)
            y = model(x)
            #y = y.cpu()
            #I = I + 1
            #print(I)
            """
            x1 = torch.squeeze(x1).numpy()
            x1 = np.transpose(x1, [1, 2, 0])
            true_mask = torch.squeeze(true_mask).numpy()
            #img_y = torch.squeeze(y).numpy()
            img_y=y.numpy()[0][0]
            """
            # print(type(x1))
            # print(type(true_mask))
            # print(type(img_y))

            # for i in range(len(img_y)):
            #     for j in range(len(img_y[0])):
            #         if img_y[i][j] > 0.5:
            #             img_y[i][j] = 1
            #         else:
            #             img_y[i][j] = 0

            # TP = 0  # True Positive,被判定为正样本，事实上也是证样本
            # TN = 0  # True Negative,被判定为负样本，事实上也是负样本
            # FP = 0  # False Positive,被判定为正样本，但事实上是负样本
            # FN = 0  # False Negative,被判定为负样本，但事实上是正样本
            #
            # for i in range(len(img_y)):
            #     for j in range(len(img_y[0])):
            #         if true_mask[i][j] > 0.5 and img_y[i][j] > 0.5:
            #             TP = TP + 1
            #         if true_mask[i][j] == 0 and img_y[i][j] == 0:
            #             TN = TN + 1
            #         if true_mask[i][j] == 0 and img_y[i][j] > 0.5:
            #             FP = FP + 1
            #         if true_mask[i][j] > 0.5 and img_y[i][j] == 0:
            #             FN = FN + 1
            #
            # P = TP / (TP + FP)
            # R = TP / (TP + FN)  # mTPR
            # ACC = (TP + TN) / (TP + TN + FP + FN)
            # IOU = TP / (TP + FP + FN)
            # FPR = FP / (FP + TN)
            # FNR = FN / (TP + FN)
            # F1 = (2 * R * P) / (R + P)
            # NICE1 = (FP + FN) / (512 * 512)
            # NICE2 = (FPR + FNR) / 2
            #
            # print('Pi:', P)
            # print('Recall:', R)
            # print('Pixel_Accuracy:', ACC)
            # print('iou:', IOU)
            # print('f1:', F1)
            # print('NICE1', NICE1)
            # print('NICE2:', NICE2)
            #
            # P_score.append(P)
            # R_scores.append(R)
            # ACC_score.append(ACC)
            # IOU_score.append(IOU)
            # F1_score.append(F1)
            # NICE1_score.append(NICE1)
            # NICE2_score.append(NICE2)

            # allim = []
            # for i in range(512):
            #     one0 = []
            #     for j in range(512):
            #         if true_mask[i][j] > 0.5 and img_y[i][j] < 0.5:
            #             data = x[0][0][i][j] + 1 * true_mask[i][j]
            #             one0.append([x[0][0][i][j], data, x[0][0][i][j]])
            #         elif true_mask[i][j] < 0.5 and img_y[i][j] > 0.5:
            #             data = x[0][0][i][j] + 1 * img_y[i][j]
            #             one0.append([data, x[0][0][i][j], x[0][0][i][j]])
            #         else:
            #             one0.append([x[0][0][i][j], x[0][1][i][j], x[0][2][i][j]])
            #     allim.append(one0)
            # allim = np.array(allim, dtype="float32")
            """
            allim1 = []
            for i in range(512):
                one0 = []
                for j in range(512):
                    aaa=img_y[i][j]
                    bbb=1/(1+math.exp(-aaa))
                    if bbb > 0.5:
                        one0.append([x[0][0][i][j] + 0.6, x[0][0][i][j] + 0.6, x[0][0][i][j]])
                    else:
                        one0.append([x[0][0][i][j], x[0][1][i][j], x[0][2][i][j]])
                allim1.append(one0)
            allim1 = np.array(allim1, dtype="float32")
            """
            #plt.figure("Image")  # 图像窗口名称
            #plt.axis('off')
            #plt.imshow(allim1, cmap=plt.cm.gray)
            # plt.show()
            # print(allim1.shape)
            #plt.savefig('000.png', bbox_inches='tight', pad_inches=-0.1)
            #plt.show()

            # fig = plt.figure(figsize=(10, 10))
            # ax0 = fig.add_subplot(1, 5, 1)
            # ax1 = fig.add_subplot(1, 5, 2)
            # plt.yticks([])
            # ax2 = fig.add_subplot(1, 5, 3)
            # plt.yticks([])
            # ax3 = fig.add_subplot(1, 5, 4)
            # plt.yticks([])
            # ax4 = fig.add_subplot(1, 5, 5)
            # plt.yticks([])
            #
            # ax0.imshow(x1)
            # ax1.imshow(true_mask, cmap=plt.cm.gray)
            # ax2.imshow(img_y, cmap=plt.cm.gray)
            # ax3.imshow(allim1, cmap=plt.cm.gray)
            # ax4.imshow(allim, cmap=plt.cm.gray)
            # plt.show()
            # plt.pause(20)

    # print("mean_P_score:", np.mean(P_score))
    # print("mean_R(mTPR)_score:", np.mean(R_scores))
    # print("mean_ACC_score:", np.mean(ACC_score))
    # print("mean_IOU_score:", np.mean(IOU_score))
    # print("mean_F1_score:", np.mean(F1_score))
    # print("mean_NICE1_score:", np.mean(NICE1_score))
    # print("mean_NICE2_score:", np.mean(NICE2_score))

    # matplotlib.image.imsave('name.png', x1)
    b = time.time()
    print("time:%d"%(b-a))

parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--batch_size", type=int, default=1)
parse.add_argument("--ckp", type=str, help="the path of model weight file")
args = parse.parse_args()

if __name__ == '__main__':
    train()  # train

    # args.ckp = "weights_99.pth"     # test()
    # test()
