#coding:utf-8
import os
import numpy as np
from PIL import Image

#重新命名
def FileReName(DogType,FilePath):
    type_counter = 0
    for type in DogType:
        file_counter = 0
        subfolder = os.listdir(FilePath+type)
        for subclass in subfolder:
            file_counter +=1
            print (file_counter)
            print ('Type_counter',type_counter)
            print (subclass)
            os.rename(FilePath+type+'/'+subclass, FilePath+type+'/'+str(type_counter)+'_'+str(file_counter)+'_'+subclass.split('.')[0]+'.jpg')
        type_counter += 1
#重新图片尺寸

def FileResize(Output_folder,DogType,FilePath,Width=100, Height=100):
    for type in DogType:
        for i in os.listdir(FilePath+type):
            img_open = Image.open(FilePath+type+'/'+i)
            conv_RGB = img_open.convert('RGB')
            Resized_img = conv_RGB.resize((Width,Height),Image.BILINEAR)
            Resized_img.save(os.path.join(Output_folder,os.path.basename(i)))

#读取图片返回array数组 numpy array
def ReadImage(filename,train_folder):
    img = Image.open (train_folder+filename)
    return np.array(img)

#图片加载到列表 图像 和 标签
def DataSet(train_folder):
    Train_list_img = []
    Train_list_label = []

    for file_1 in os.listdir(train_folder):
        file_img_to_array = ReadImage(filename=file_1,train_folder=train_folder)
        #添加图片数组到主list里
        Train_list_img.append(file_img_to_array)
        # 添加标签数组到主list里
        Train_list_label.append(int(file_1.split('_')[0]))

    Train_list_img = np.array(Train_list_img)
    Train_list_label = np.array(Train_list_label)

    print(Train_list_img.shape)   #X_train minst
    print(Train_list_label.shape)
    print(Train_list_img,Train_list_label)#Y_train minst



if __name__ == "__main__":

    DogType = ['哈士奇','德国牧羊犬','拉布拉多','萨摩耶犬']
    #修改名字

    #FileReName(DogType=DogType,FilePath='Raw_Img/')

    #修改尺寸
    FileResize(DogType=DogType, FilePath='Raw_img/',Output_folder='train_img/')

    #准备好的数据
    #DataSet(train_folder='train_img/')

    # FileReName(DogType=DogType,FilePath='Raw_Img/')

    #修改尺寸
    # FileResize(DogType=DogType, FilePath='Raw_Img/',Output_folder='train_img/')

    #准备好的数据
    DataSet(train_folder='train_img/')



