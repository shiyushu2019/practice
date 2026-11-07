import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from torchplus.utils import ClassificationAccuracy
from torchplus.models import resnet50fe, ResNet50FE_Dim
from utils.model import CLS

if __name__ == "__main__":
    batch_size = 64
    train_epoches = 48
    log_epoch = 4
    seed = 9970
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # root_dir = "./log/paper5w/logZZPMAIN.resnetface"
    # 训练数据来自 FaceScrub，统一 resize 到 224x224，并按 6:1 的比例切分训练/测试集。
    dataset_dir = "/data1/dataset/jssnu005/sys/lh/FaceScrub"
    h = 128
    w = 128

    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "./checkpoints/resnetface"
    log_dir = "./checkpoints/resnetface"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    data_workers = 2

    from utils.datasplit import (
            ds,
            part1_ds,
            test_ds
    )
    train_ds=part1_ds
    class_num = len(ds.classes)
    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)
    print(train_ds_len)
    print(test_ds_len)
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )
    train_dl1 = DataLoader(  #用训练集评估时候用的
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )
    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    feature_extractor = (
        resnet50fe(weights=ResNet50_Weights.IMAGENET1K_V2).train(True).to(output_device)
    )
    cls = CLS(ResNet50FE_Dim, class_num).train(True).to(output_device)

    optimizer_fe = optim.Adam(
        feature_extractor.parameters(), lr=0.00002, betas=(0.5, 0.999), amsgrad=True
    )

    optimizer_cls = optim.Adam(
        cls.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    # 训练主循环：先前向计算特征和 logits，再用交叉熵更新分类器和特征提取器；
    # 每隔若干轮记录损失和准确率，并保存模型快照。
    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch", mininterval=1.0):
        feature_extractor.train()
        cls.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        iters = tqdm(train_dl, desc=f"epoch {epoch_id}", mininterval=1.0)
        for i, (im, label) in enumerate(iters):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer_fe.zero_grad()
            optimizer_cls.zero_grad()
            feature = feature_extractor.forward(im)
            out = cls.forward(feature)
            ce = nn.CrossEntropyLoss()(out, label)
            loss = ce
            loss.backward()
            optimizer_cls.step()
            optimizer_fe.step()
            batch_samples = label.size(0)
            epoch_loss += loss.detach().item() * batch_samples
            epoch_correct += (out.argmax(dim=-1) == label).sum().item()
            epoch_samples += batch_samples

        if epoch_id % log_epoch == 0:
            print("loss", epoch_loss / epoch_samples, epoch_id)
            print("acc_training", epoch_correct / epoch_samples, epoch_id)
            with open(
                os.path.join(log_dir, f"feature_extractor_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(feature_extractor.state_dict(), f)
            with open(os.path.join(log_dir, f"cls_{epoch_id}.pkl"), "wb") as f:
                torch.save(cls.state_dict(), f)

            with torch.no_grad():
                feature_extractor.eval()
                cls.eval()
                train_samples = 0
                celoss = 0.0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(train_dl1, desc="testing train", mininterval=1.0)):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature = feature_extractor.forward(im)
                    out = cls.forward(feature)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    batch_samples = label.size(0)
                    celoss += ce.item() * batch_samples
                    train_samples += batch_samples

                celossavg = celoss / train_samples
                acc_test = test_ca.get()
                print("acc_train", acc_test, epoch_id)
                print("train loss", celossavg, epoch_id)

                test_samples = 0
                celoss = 0.0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(test_dl, desc="testing test", mininterval=1.0)):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature = feature_extractor.forward(im)
                    out = cls.forward(feature)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    batch_samples = label.size(0)
                    celoss += ce.item() * batch_samples
                    test_samples += batch_samples

                celossavg = celoss / test_samples
                acc_test = test_ca.get()
                print("test loss", celossavg, epoch_id)
                print("acc_test", acc_test, epoch_id)
