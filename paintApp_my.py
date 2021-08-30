from tkinter import *
from PIL import ImageDraw, Image, ImageGrab
import numpy as np
#import scipy.misc
from skimage import color
from skimage import io
from keras.models import Sequential, model_from_json
import os
import io
from keras_cnn import trainModel, getData

class Paint(object):

    def __init__(self):
        self.root = Tk()

        #defining Canvas
        self.c = Canvas(self.root, bg='white', width=280, height=280)
        
        self.image1 = Image.new('RGB', (280, 280), color = 'white')
        self.draw = ImageDraw.Draw(self.image1) 

        self.c.grid(row=1, columnspan=5)

        self.classify_button = Button(self.root, text='辨識', command=lambda:self.classify(self.c))
        self.classify_button.grid(row=0, column=1)

        self.clear = Button(self.root, text='清畫面', command=self.clear)
        self.clear.grid(row=0, column=3)

        self.prediction_text = Text(self.root, height=2, width=10)
        self.prediction_text.grid(row=2, column=3)

        self.model = self.loadModel()
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = 'black'
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            # 畫圖同時寫到記憶體，避免螢幕字型放大，造成抓到的畫布區域不足
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill='black')

        self.old_x = event.x
        self.old_y = event.y

    def clear(self):
        self.c.delete("all")
        self.image1 = Image.new('RGB', (280, 280), color = 'white')
        self.draw = ImageDraw.Draw(self.image1) 
        self.prediction_text.delete("1.0", END)

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
    def classify(self, widget):
        # 顯示設定>100%，會造成辨識不佳，因為抓到的區域會變小
        #getting pixel information
        # x=self.root.winfo_rootx()+widget.winfo_x()
        # y=self.root.winfo_rooty()+widget.winfo_y()
        # x1=x+widget.winfo_width()
        # y1=y+widget.winfo_height()
        # ImageGrab.grab().crop((x,y,x1,y1)).resize((28, 28)).save('classify.png')
        #save drawing
        #img = scipy.misc.imread('classify.png', flatten=False, mode='P')
        
        # widget.postscript(file = './classify.eps') 
        # # use PIL to convert to PNG 
        # img = ImageGrab.Image.open('./classify.eps') 
        # img.save('./classify.png')
        # img = color.rgb2gray(io.imread('./classify.png')).resize( (28, 28), ImageGrab.Image.ANTIALIAS)
        
        # img = color.rgb2gray(self.image1.resize( (28, 28), ImageGrab.Image.ANTIALIAS))
        img = self.image1.resize((28, 28), ImageGrab.Image.ANTIALIAS).convert('L')
        img.save('1.png')
        
        img = np.array(img)
        # Change pixels to work with our classifier
        # img[img==0] = 255
        # img[img==225] = 0
        img[img==255] = 0
        img[img>100] = 255
        
        img2=Image.fromarray(img) 
        img2.save('2.png')

        img = np.reshape(img, (1, 28, 28, 1))
        
        # Predict digit
        pred = self.model.predict([img])
        # Get index with highest probability
        pred = np.argmax(pred)
        print(pred)
        self.prediction_text.delete("1.0", END)
        self.prediction_text.insert(END, pred)

    def loadModel(self):
        if(os.path.exists('mnist_model.h5')):
            print('loading model')
            json_file = open('model.json', 'r')
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json)
            model.load_weights("mnist_model.h5")
            return model
        else:
            print('train model')
            X_train, y_train, X_test, y_test = getData()
            model = trainModel(X_train, y_train, X_test, y_test)
            return model

if __name__ == '__main__':
    Paint()
