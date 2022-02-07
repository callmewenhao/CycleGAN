import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class A2BDataset(Dataset):
    def __init__(self, A_dir, B_dir, transform=None):
        super(A2BDataset, self).__init__()
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.transform = transform

        self.A_images = os.listdir(A_dir)
        self.B_images = os.listdir(B_dir)
        self.A_len = len(self.A_images)  # 注意AB两个文件夹中的图片数量不一样
        self.B_len = len(self.B_images)

    def __len__(self):
        return max(self.A_len, self.B_len)

    def __getitem__(self, item):
        A_image_name = self.A_images[item % self.A_len]
        B_image_name = self.B_images[item % self.B_len]
        A_image_path = os.path.join(self.A_dir, A_image_name)
        B_image_path = os.path.join(self.B_dir, B_image_name)

        A_image = Image.open(A_image_path).convert('RGB')
        B_image = Image.open(B_image_path).convert('RGB')

        if self.transform:
            A_image = self.transform(A_image)
            B_image = self.transform(B_image)

        return A_image, B_image



def main():
    # path
    A_dir = "G:\CycleGAN_DATA\\apple2orange\\trainA"  # 995
    B_dir = "G:\CycleGAN_DATA\\apple2orange\\trainB"  # 1019

    # transform
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # dataset
    Apple2OrangeDatset = A2BDataset(A_dir, B_dir, transform)
    print(len(Apple2OrangeDatset))
    A_image = Apple2OrangeDatset[0][0]
    B_image = Apple2OrangeDatset[0][1]
    print(A_image.shape, B_image.shape)


if __name__ == "__main__":
    main()


