from torch.utils.data import DataLoader, Dataset

class IQ_data(Dataset):
  def __init__(self, data_x, data_y, transforms=None):
    super().__init__()

    self.data_x = data_x
    self.data_y = data_y
    self.transforms = transforms

  def __len__(self):
    return self.data_x.shape[0]

  def __getitem__(self, idx):
    datapoint = self.data_x[idx]

    if self.transforms is not None:
      datapoint = self.transforms(datapoint)

    datapoint = torch.Tensor(datapoint)
    datapoint = datapoint.permute(2,0,1)

    if self.data_y is not None:
      label = self.data_y[idx]
      return datapoint, label
    
    return datapoint



class IQ_data_two(Dataset): 
