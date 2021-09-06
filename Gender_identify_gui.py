# Gender_identify_gui.py
# 图形化界面
import torch
from tkinter import *
from random import choice
from PIL import Image, ImageTk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from YutongNet2 import YutongNet

# 获取名字的集合
f = open("names.txt")
names = f.readlines()
f.close()

# 分别获取男女名单，用于标注label
f_f = open("female_names.txt")  # 打开女名文件
f_name = f_f.readlines()  # 将女名读入list
f_f.close()  # 关闭文件

m_f = open("male_names.txt")  # 打开男名文件
m_name = m_f.readlines()  # 将男名读入list
m_f.close()  # 关闭文件


class SimpleDataset(Dataset):
    def __init__(self, name_file, transform=None):
        self.name_file = name_file
        self.transform = transform

        file_path = 'lfw_funneled/' + self.name_file[:-9] + '/' + self.name_file
        image = Image.open(file_path)
        if self.transform:
            self.image = self.transform(image)

    def __getitem__(self, index):

        if self.name_file + '\n' in f_name:
            label = 0
        else:
            label = 1

        return self.image, label

    def __len__(self):
        return 1


def show(p_name):

    global p_img, p_photo, p_img_label

    p_name_path = 'lfw_funneled/' + p_name[:-9] + '/' + p_name
    if not os.path.isfile(p_name_path):
        print(p_name + 'does not exist!')
    else:
        p_img = Image.open(p_name_path)
        p_photo = ImageTk.PhotoImage(p_img)
        p_img_label = Label(root, image=p_photo)
        p_img_label.place(x=90, y=80)

        p_name_label = Label(root, text=p_name[:-9], font=20, bg='white', width=30)
        p_name_label.place(anchor=CENTER, x=215, y=380)

        simple_data = SimpleDataset(name_file=p_name,
                                    transform=transforms.Compose([
                                        # transforms.Resize((128, 128)),
                                        # transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.Pad(10),
                                        # transforms.RandomCrop((100, 100)),
                                        transforms.CenterCrop(150),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])]))
        simple_loader = DataLoader(simple_data, batch_size=1)

        for ph, target in simple_loader:
            ph = ph.to(device)
            target = target.to(device)

            with torch.no_grad():
                outputs = model(ph)

            if outputs.argmax(1) == 1:
                guess = 'Man, right?'
            else:
                guess = 'Woman, right?'

            guess_label = Label(root, text=guess, font=25, width=20, bg='white')
            guess_label.place(anchor=CENTER, x=550, y=100)


def rand_show():
    # 该函数用于Random Draw被点击时
    rand_name = choice(names)[:-1]
    show(rand_name)


def appoint_show():
    pic = t.get()
    if pic + '\n' not in names:
        error_label = Label(root, text="No such picture!", font=25, width=20, bg='white')
        error_label.place(anchor=CENTER, x=550, y=100)
    else:
        show(pic)


# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YutongNet().to(device)
model = torch.load("best_model.pth")
model.eval()


if __name__ == '__main__':
    # 初始化GUI
    root = Tk()
    root.title("Gender Identify")
    root.geometry('800x500')

    ran_label = Button(root, text='Random Draw', command=rand_show)
    ran_label.place(anchor=CENTER, x=550, y=150)

    name = choice(names)[:-1]
    name_path = 'lfw_funneled/' + name[:-9] + '/' + name
    p_img = Image.open(name_path)
    p_photo = ImageTk.PhotoImage(p_img)
    p_img_label = Label(root, image=p_photo)
    p_img_label.place(x=90, y=80)

    hint_label = Label(root, text='You could enter a picture name in Lwf dataset\n\
    For example: Norman_Mailer_0001.jpg', anchor=W)
    hint_label.place(anchor=CENTER, x=550, y=250)

    t = Entry(root, width=40)
    t.place(anchor=CENTER, x=550, y=300)

    t_label = Button(root, text='Predict gender!', command=appoint_show)
    t_label.place(anchor=CENTER, x=550, y=350)

    root.mainloop()
