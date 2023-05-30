import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# trick: transfer the data centre to zero and multiply a scale
def pred_transform(pred):
    pred = (1 / 6.0) * pred
    pred += 12.0
    pred = torch.exp(pred)

    return pred

class KaggleHouse:
    def preprocess(self):
        label_name = 'SalePrice'
        features = pd.concat((self.raw_train_data.drop(columns=['Id', label_name]),
                              self.raw_test_data.drop(columns=['Id'])))
        numeric_features = features.dtypes[features.dtypes != 'object'].index
        features[numeric_features] = features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
        features[numeric_features] = features[numeric_features].fillna(0)
        features = pd.get_dummies(features, dummy_na=True)

        train_data = features[:self.raw_train_data.shape[0]].copy()
        label = self.raw_train_data[label_name]
        test_data = features[self.raw_train_data.shape[0]:].copy()

        return train_data, label, test_data
    
    def __init__(self, train_data_path, test_data_path):
        self.raw_train_data = pd.read_csv(train_data_path)
        self.raw_test_data = pd.read_csv(test_data_path)
        self.train_data, self.label, self.test_data = self.preprocess()
    
    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data, self.raw_test_data['Id']
    
    def get_label(self):
        return self.label
    
    def k_fold(self, k):
        data_size = len(self.label)
        fold_size = data_size // k
        for i in range(k):
            valid_idx = np.array(range(i*fold_size, min((i+1)*fold_size, data_size)))
            train_idx = np.delete(range(data_size), valid_idx)
            yield train_idx, valid_idx

class KaggleHouseDataset(Dataset):
    def label_transform(self):
        # log label to avoid price numerical gap influence
        self.label = torch.log(self.label)
        self.label -= 12.0
        self.label = 6 * self.label
        return self.label

    def __init__(self, feature, label):
        super(KaggleHouseDataset, self).__init__()
        self.feature = torch.tensor(np.array(feature).astype(np.float32))
        self.label = torch.Tensor(np.array(label))
        self.label = self.label_transform()
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]

class KaggleHouseModel(nn.Module):
    def __init__(self):
        super(KaggleHouseModel, self).__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(200),
            nn.ReLU(),
            nn.LazyLinear(50),
            nn.ReLU(),
            nn.LazyLinear(1))

    def forward(self, x):
        return self.net(x)

# load data
kaggle_house_data = KaggleHouse("./Data/train.csv", "./Data/test.csv")
kaggle_house_dataset = KaggleHouseDataset(kaggle_house_data.get_train_data(),
                                          kaggle_house_data.get_label())

# general settings
device = "cuda" if torch.cuda.is_available() else "cpu"
k = 10
num_epoch = 500
lr = 0.015

# train
# for k-fold-validation
fold_num = 0
for train_idx, valid_idx in kaggle_house_data.k_fold(k):
    fold_num += 1
    print("\nFold ", fold_num, " ")

    train_data_subsampler = SubsetRandomSampler(train_idx)
    valid_data_subsampler = SubsetRandomSampler(valid_idx)
    train_dataloader = DataLoader(kaggle_house_dataset,
                                  batch_size=16,
                                  sampler=train_data_subsampler)
    valid_dataloader = DataLoader(kaggle_house_dataset,
                                  batch_size=16,
                                  sampler=valid_data_subsampler)
    
    model = KaggleHouseModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    
    # for epoch
    for i in range(num_epoch):
        # for minibatch
        train_loss = 0.0
        train_num = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            output = torch.reshape(output, (-1,))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.mean().item() * y.size(0)
            train_num += y.size(0)
        
        print("Epoch ", i, ", train loss ", train_loss / train_num)
        
    with torch.no_grad():
        valid_loss = 0.0
        valid_num = 0
        for x_v, y_v in valid_dataloader:
            x_v = x_v.to(device)
            y_v = y_v.to(device)

            output_v = model(x_v)
            output_v = torch.reshape(output_v, (-1,))
            loss_v = criterion(output_v, y_v)

            valid_loss += loss_v.mean().item() * y_v.size(0)
            valid_num += y_v.size(0)
    print("Fold ", fold_num, ", end, avg_log_loss: ", valid_loss / valid_num)

    with torch.no_grad():
        test_data, test_id = kaggle_house_data.get_test_data()
        test_data = torch.tensor(np.array(test_data).astype(np.float32)).to(device)
        output_t = model(test_data)
        output_t = pred_transform(output_t)
        output_t = output_t.cpu().numpy()
        
        output_df = pd.DataFrame(output_t, columns=['SalePrice'])
        df = pd.concat([test_id, output_df], axis=1)
        save_file = './outputs/res_model_' + str(fold_num) + '.csv'
        df.to_csv(save_file, index=False)
        
