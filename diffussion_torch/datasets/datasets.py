from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from tqdm import tqdm
import cv2

class FlowsDataset(Dataset.Dataset):
    def __init__(self, txt_path = "flowers_data.txt"):
        with open(txt_path, "r") as f:
            data = f.readlines()
        self.image_path = data[:-1]

        self.transforms = transforms.Compose([
                transforms.Resize(80),
                transforms.RandomResizedCrop(64,scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    def __getitem__(self, index):
        image_path , label = self.image_path[index].split(";")
        image = Image.open(image_path) 
        image = self.transforms(image)
        
        #print("image.numpy(): ",image)
        # cv2.imshow("asd",image.numpy())
        # cv2.waitKey(0)


        label = int(label)
        return image, label
    
    def __len__(self):
        return len(self.image_path)
    

    
if __name__ == "__main__": # 测试数据集是否可用
    dataset = FlowsDataset()
    dataloader = DataLoader.DataLoader(dataset, batch_size=24, shuffle=True)

    pbar = tqdm(dataloader)
    for i, (images, labels) in enumerate(pbar):
        pass
    print("ok")

