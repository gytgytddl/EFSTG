import numpy as np
import os
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    print(f"正在从 {dataset_dir} 加载数据...")
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
    print("=" * 50)
    train_data_non_overlapping = data["x_train"][:, -1, :, 0]
    data_mean = np.mean(train_data_non_overlapping)
    data_std = np.std(train_data_non_overlapping)
    print(f"标准化参数: Mean: {data_mean:.6f}, Std: {data_std:.6f}")
    scaler = StandardScaler(mean=data_mean, std=data_std)
    data["scaler"] = scaler
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
        data["y_" + category] = scaler.transform(data["y_" + category])

    def create_torch_dataloader(x_numpy, y_numpy, batch_size, shuffle, num_workers=4, drop_last=False, seed=42):
        x_tensor = torch.FloatTensor(x_numpy)
        y_tensor = torch.FloatTensor(y_numpy)
        dataset = TensorDataset(x_tensor, y_tensor)
        g = torch.Generator()
        g.manual_seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=drop_last,
            generator=g if shuffle else None
        )
        return loader

    class DataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader

        def get_iterator(self):
            return iter(self.loader)

        def shuffle(self):
            pass

        def __len__(self):
            return len(self.loader)

    print("构建 PyTorch DataLoader (开启 pin_memory 和多线程)...")
    train_loader = create_torch_dataloader(data["x_train"], data["y_train"], batch_size, shuffle=True, drop_last=True)
    val_loader = create_torch_dataloader(data["x_val"], data["y_val"], valid_batch_size, shuffle=False, drop_last=False)
    test_loader = create_torch_dataloader(data["x_test"], data["y_test"], test_batch_size, shuffle=False,
                                          drop_last=False)
    data["train_loader"] = DataLoaderWrapper(train_loader)
    data["val_loader"] = DataLoaderWrapper(val_loader)
    data["test_loader"] = DataLoaderWrapper(test_loader)
    print(f"数据加载完成")
    return data


def masked_mae_loss(preds, labels, null_val=0.0):
    if torch.isnan(preds).any():
        preds = torch.nan_to_num(preds)
    mask = (labels > null_val).float()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / torch.clamp(torch.sum(mask), min=1.0)

def masked_huber_loss(preds, labels, null_val=0.0, delta=1.0):
    if torch.isnan(preds).any():
        preds = torch.nan_to_num(preds)
    mask = (labels > null_val).float()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = F.huber_loss(preds, labels, reduction='none', delta=delta)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / torch.clamp(torch.sum(mask), min=1.0)

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    if torch.isnan(pred).any():
        pred = torch.nan_to_num(pred, nan=0.0)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=1.0):
    if mask_value is not None:
        mask = torch.ge(true, mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def WMAPE_torch(pred, true, mask_value=None):
    if (mask_value != None):
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def metric(pred, true):
    if pred.shape[0] != true.shape[0]:
        min_len = min(pred.shape[0], true.shape[0])
        pred = pred[:min_len]
        true = true[:min_len]
    mae = MAE_torch(pred, true, 0.0).item()
    mape = MAPE_torch(pred, true, 1.0).item()
    rmse = RMSE_torch(pred, true, 0.0).item()
    wmape = WMAPE_torch(pred, true, 0.0).item()
    return [mae, mape, rmse, wmape]