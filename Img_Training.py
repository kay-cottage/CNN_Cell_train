#coding:utf-8

import os
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D,Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


#Pre process images
class PreFile(object):
    def __init__(self,FilePath,Celltype):
        self.FilePath = FilePath
        self.Celltype = Celltype
        

    def FileReName(self):
        count = 0
        for type in self.CellType: #For dog type output each dog foler name
            subfolder = os.listdir(self.FilePath+type)  # list up all folder
            for subclass in subfolder:  #output name of folder
                print ('count_classese:->>' , count)
                print (subclass)
                print (self.FilePath+type+'/'+subclass)
                os.rename(self.FilePath+type+'/'+subclass, self.FilePath+type+'/'+str(count)+'_'+subclass.split('.')[0]+".jpg")
            count+=1

    def FileResize(self,Width,Height,Output_folder):
        for type in self.CellType:
            print (type)
            files = os.listdir(self.FilePath+type)
            for i in files:
                img_open = Image.open(self.FilePath + type+'/' + i)
                conv_RGB = img_open.convert('RGB') #统一转换一下RGB格式 统一化
                new_img = conv_RGB.resize((Width,Height),Image.BILINEAR)
                new_img.save(os.path.join(Output_folder,os.path.basename(i)))

#main Training program
class Training(object):
    def __init__(self,batch_size,number_batch,categories,train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    #Read image and return Numpy array
    def read_train_images(self,filename):
        img = Image.open(self.train_folder+filename)
        return np.array(img)

    def train(self):
        train_img_list = []
        train_label_list = []
        for file in os.listdir(self.train_folder):
            files_img_in_array =  self.read_train_images(filename=file)
            train_img_list.append(files_img_in_array) #Image list add up
            train_label_list.append(int(file.split('_')[0]))
        print(train_label_list)#lable list addup
        train_label_list = to_categorical(train_label_list,4)
        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)
        print(train_label_list)

        #train_label_list = to_categorical(train_label_list,4) #format into binary [0,0,0,0,1,0,0]
        

        train_img_list = train_img_list.astype('float32')
        train_img_list /= 255

        #-- setup Neural network CNN
        model = Sequential()
        #CNN Layer - 1
        model.add(Convolution2D(
            filters=32, #Output for next later layer output (100,100,32)
            kernel_size= (5,5) , #size of each filter in pixel
            padding= 'same', #边距处理方法 padding method
            input_shape=(100,100,3) , #input shape ** channel last(TensorFlow)
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2,2), #Output for next layer (50,50,32)
            strides=(2,2),
            padding='same',
        ))

        #CNN Layer - 2
        model.add(Convolution2D(
            filters=64,  #Output for next layer (50,50,64)
            kernel_size=(2,2),
            padding='same',
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(  #Output for next layer (25,25,64)
            pool_size=(2,2),
            strides=(2,2),
            padding='same',
        ))

    #Fully connected Layer -1
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
    # Fully connected Layer -2
        model.add(Dense(512))
        model.add(Activation('relu'))
    # Fully connected Layer -3
        model.add(Dense(256))
        model.add(Activation('relu'))
    # Fully connected Layer -4
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))
    # Define Optimizer
        adam = Adam(lr = 0.0001)
    #Compile the model
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=['accuracy']
                      )
    # Fire up the network
        model.fit(
            train_img_list,
            train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            verbose=1,
        )
        #SAVE your work -model
        model.save('./cellfinder.h5')

     
    
def MAIN():

    CellType = ['嗜酸性粒细胞','嗜碱性粒细胞','中性粒细胞']

    #Trainning Neural Network
    Train = Training(batch_size=128,number_batch=60,categories=3,train_folder='train_img/')
    Train.train()


if __name__ == "__main__":
    MAIN()





