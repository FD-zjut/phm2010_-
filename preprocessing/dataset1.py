import torch
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, data_row):
        self.data_row_dataset = torch.tensor(data_row[:, 2:12])
        self.data_row_dataset=torch.unsqueeze(self.data_row_dataset, 2)
            
        self.label = torch.tensor(data_row[:, 1])
        self.label=torch.unsqueeze(self.label, 1)
            
    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data_row_dataset[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data_row_dataset)
