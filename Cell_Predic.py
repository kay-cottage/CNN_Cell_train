#coding:utf-8

from tensorflow.keras.models import load_model
import matplotlib.image as processimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Prediction(object):
    def __init__(self,ModelFile,PredictFile,CellType,Width=100,Height=100):
        self.modelfile = ModelFile
        self.predict_file = PredictFile
        self.Width = Width
        self.Height = Height
        self.CellType = CellType

    def Predict(self):
        #加载预测照片并且将其处理成100*100大小的图像
        model = load_model(self.modelfile)
        img_open = Image.open(self.predict_file)
        conv_RGB = img_open.convert('RGB')
        new_img = conv_RGB.resize((self.Width,self.Height),Image.BILINEAR)
        new_img.save(self.predict_file)
        print('Image Processed')

        image = processimage.imread(self.predict_file)
        image_to_array = np.array(image)/255.0
        image_to_array = image_to_array.reshape(-1,100,100,3)
        print('Image reshaped')

        prediction = model.predict(image_to_array)
        print(prediction)
        Final_prediction = [result.argmax() for result in prediction][0]
        print(Final_prediction)

        #打印出预测结果
        count = 0
        for i in prediction[0]:
            print(i)
            percentage = '%3f%%' % (i * 100)
            print(self.DogType[count],'概率:' ,percentage)
            count +=1

        #展示导入的预测图象
    def ShowPredImg(self):
        image = processimage.imread(self.predict_file)
        plt.imshow(image)
        plt.show()


CellType = ['嗜酸性粒细胞','嗜碱性粒细胞','中性粒细胞'，'空']

#其中PredictFile='2.jpg'为需要识别的图像,ModelFile='cellfinder.h5'为之前训练好的模型文件,CellType为上面几种细胞分类集中的种类列表
Pred = Prediction(PredictFile='2.jpg',ModelFile='cellfinder.h5',Width=100,Height=100,CellType=CellType)
Pred.Predict()


