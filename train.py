import os
import torch
import copy
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from models.unet import UNet_conditional
from utils.ema import EMA
import logging
import numpy as np
from diffusion import Diffusion
import warnings
import argparse
import torch.utils.data.dataloader as DataLoader
from datasets.datasets import FlowsDataset
from utils.utils import save_images

def setup_logging():
    # 创建一个记录器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # 创建一个处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建一个格式化器并将其添加到处理器中
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # 将处理器添加到记录器中
    logger.addHandler(ch)

    return logger


def train(args):
    logger_my = setup_logging()
    device = args.device

    dataset = FlowsDataset(txt_path = args.dataset_txt)
    dataloader = DataLoader.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    for epoch in range(args.epochs):
        logger_my.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            
            loss = mse(noise, predicted_noise)  # 损失函数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
 
        if epoch % 10 == 0:     # 保存模型，可视化训练结果。
            torch.save(model.state_dict(), f"{args.checkpoint_save}/ddpm_cond{epoch}.pdparams")

            labels = torch.arange(5,dtype=int).to(device)
            sampled_images1 = diffusion.sample(model, n=len(labels), labels=labels)
            sampled_images2 = diffusion.sample(model, n=len(labels), labels=labels)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

            sampled_images1 = sampled_images1.cpu().numpy()
            sampled_images2 = sampled_images2.cpu().numpy()

            for i in range(5):
                img = sampled_images1[i].transpose([1, 2, 0])
                img = np.array(img).astype("uint8")
                plt.subplot(2,5,i+1)
                plt.imshow(img)
            for i in range(5):
                img = sampled_images2[i].transpose([1, 2, 0])
                img = np.array(img).astype("uint8")
                plt.subplot(2,5,i+1+5)
                plt.imshow(img)
            plt.savefig(f"./logs/images/{epoch}.png")

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description="train diffusion models")
    parser.add_argument("--run_name",default="DDPM_Uncondtional",required=False,type=str,help="Experience name.")
    parser.add_argument("--epochs", default=300, required=False,type=int, help=" ")
    parser.add_argument("--batch_size", default=8, required=False,type=int, help="Batch size.")
    parser.add_argument("--image_size",default=64,required=False,type=int, help="how large your image.")
    parser.add_argument("--device",default="cuda",required=False,help="train device.")
    parser.add_argument("--lr",default=1.5e-4,required=False,type=float, help="learning rate.")
    parser.add_argument("--num_classes",default=5,required=False,type=int,help="how much class.")
    parser.add_argument("--dataset_txt",default="./data/flowers_data.txt",required=False,type=str,help="")
    parser.add_argument("--checkpoint_save",default="./weights",required=False,type=str,help="")
    args = parser.parse_args()

    train(args=args)
