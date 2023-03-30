#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true
import cv2
import imutils
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from tkinter import *
  

def show_img(img):
   fig = plt.gcf()
   fig.set_size_inches(16, 8)
   plt.axis("off")
   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#   plt.show()

#img = cv2.imread('C:/Users/niko-_000/Desktop/All/pastas/academico/IA/PlacasCar/image/car1.jpg')
#(H, W) = img.shape[:2]
#print(H, W)

def detect_plate(file_img):
  img = cv2.imread(file_img)
  (H, W) = img.shape[:2]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.bilateralFilter(gray, 11, 17, 17)
  edged = cv2.Canny(blur, 30, 200)
#  show_img(edged)
  conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  conts = imutils.grab_contours(conts) 
  conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8] 

  location = None
  for c in conts:
    peri = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
    if cv2.isContourConvex(aprox):
      if len(aprox) == 4:
          location = aprox
          break

  beginX = beginY = endX = endY = None
  if location is None:
    plate = False
  else:
    mask = np.zeros(gray.shape, np.uint8) 

    img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
    img_plate = cv2.bitwise_and(img, img, mask=mask)

    (y, x) = np.where(mask==255)
    (beginX, beginY) = (np.min(x), np.min(y))
    (endX, endY) = (np.max(x), np.max(y))

    plate = gray[beginY:endY, beginX:endX]
    #show_img(plate)

  return img, plate, beginX, beginY, endX, endY

def ocr_plate(plate):
  pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
  config_tesseract = "--tessdata-dir tessdata --psm 6"
  #config_tesseract = "C:/Users/niko-_000\Desktop/All/pastas/academico/IA/PlacasCar/tessdata/por.traineddata"
  text = pytesseract.image_to_string(plate, lang="por", config=config_tesseract)
  text = "".join(c for c in text if c.isalnum())
  return text

def recognize_plate(file_img):
  img, plate, beginX, beginY, endX, endY = detect_plate(file_img)
  
  if plate is False:
    print("It was not possible to detect!")
    return 0

  text = ocr_plate(plate)
  texto_placa['text'] = text
 # img = cv2.putText(img, text, (beginX, beginY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150,255,0), 2, lineType=cv2.LINE_AA)
  #img = cv2.rectangle(img, (beginX, beginY), (endX, endY), (150, 255, 0), 2)
  #show_img(img)

 # return img, plate
#img, plate = recognize_plate('C:/Users/niko-_000/Desktop/All/pastas/academico/IA/PlacasCar/image/car1.jpg')
#recognize_plate('C:/Users/niko-_000/Desktop/All/pastas/academico/IA/PlacasCar/image/car1.jpg')
#def digita():
   #carro = input(lo)
                  
def pesquisa():
    lo = local.get()
    #carro = input(lo)
    recognize_plate(lo)
    

                  
janela = Tk()
janela.title('Detector de placas')

texto = Label(janela, text='Digite o local da imagem a baixo')

texto.grid(column=0, row=0, padx=10, pady=10)
label_local = Label(janela, text='Digite aqui')
local = Entry (janela,width=20)
local.grid(column=1, row=2)


botao = Button(janela, text='OK', command=pesquisa)

botao.grid(column=0, row=3, padx=10, pady=10)

texto_placa = Label(janela, text='')

texto_placa.grid(column=0, row=2, padx=10, pady=10)

janela.mainloop()