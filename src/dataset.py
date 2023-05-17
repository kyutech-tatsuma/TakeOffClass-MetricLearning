from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, index):
            img_name = self.df.iloc[index, 0]
            image = Image.open(img_name)

            if self.transform is not None:
                image = self.transform(image)

            label = self.df.iloc[index, 1]

            return image, label