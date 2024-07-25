# 加载数据集
"""由于条件生成需要同时提供图片标签，因此我们这里自定义数据集"""

# 1、将图片数据写入txt文件。flowers本来是分类数据集，这里我们把他的训练集和验证集都提取出来，当作我们生成模型的训练集。
import os
train_sunflower = os.listdir("./data/flowers/pic/train/sunflower")            # 0——向日葵
valid_sunflower = os.listdir("./data/flowers/pic/validation/sunflower")       # 0——向日葵
train_rose      = os.listdir("./data/flowers/pic/train/rose")                 # 1——玫瑰
valid_rose      = os.listdir("./data/flowers/pic/validation/rose")            # 1——玫瑰
train_tulip     = os.listdir("./data/flowers/pic/train/tulip")                # 2——郁金香
valid_tulip     = os.listdir("./data/flowers/pic/validation/tulip")           # 2——郁金香
train_dandelion = os.listdir("./data/flowers/pic/train/dandelion")            # 3——蒲公英
valid_dandelion = os.listdir("./data/flowers/pic/validation/dandelion")       # 3——蒲公英
train_daisy     = os.listdir("./data/flowers/pic/train/daisy")                # 4——雏菊
valid_daisy     = os.listdir("./data/flowers/pic/validation/daisy")           # 4——雏菊

with open("flowers_data.txt", 'w') as f:
    for image in train_sunflower:
        f.write("./data/flowers/pic/train/sunflower/" + image + ";" + "0" + "\n")
    for image in valid_sunflower:
        f.write("./data/flowers/pic/validation/sunflower/" + image + ";" + "0" + "\n")
    for image in train_rose:
        f.write("./data/flowers/pic/train/rose/" + image + ";" + "1" + "\n")
    for image in valid_rose:
        f.write("./data/flowers/pic/validation/rose/" + image + ";" + "1" + "\n")
    for image in train_tulip:
        f.write("./data/flowers/pic/train/tulip/" + image + ";" + "2" + "\n")
    for image in valid_tulip:
        f.write("./data/flowers/pic/validation/tulip/" + image + ";" + "2" + "\n")
    for image in train_dandelion:
        f.write("./data/flowers/pic/train/dandelion/" + image + ";" + "3" + "\n")
    for image in valid_dandelion:
        f.write("./data/flowers/pic/validation/dandelion/" + image + ";" + "3" + "\n")
    for image in train_daisy:
        f.write("./data/flowers/pic/train/daisy/" + image + ";" + "4" + "\n")
    for image in valid_daisy:
        f.write("./data/flowers/pic/validation/daisy/" + image + ";" + "4" + "\n")

f.close()