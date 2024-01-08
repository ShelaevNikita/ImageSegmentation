
import json
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from random import shuffle
from time import thread_time

from skimage import draw
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte

class Label2Image():
    
    CONFIGKEYS = [ 'dataImages',  'dataLabels',  'grayImages', 'gray2Colors', 'splitConfig',
                  'labelImages', 'blendImages',  'blendConfig', 'splitTrain',   'splitTest']

    def __init__(self, pathData, configFileName):
        self.pathData       = pathData
        self.configFileName = configFileName   

        self.configDict  = {}   
        self.labelDict   = {}
        
        self.labelImage  = np.zeros(1)

        self.grayColor   = []
        self.frameIDList = []
    
    def diplayImage(self, image, cmap = None):
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_subplot()
        ax.imshow(image, cmap)
        plt.show()
        return

    def saveImage(self, fileName, image):
        imsave(fileName, image, None, False)
        return

    def saveSplitData(self, splitFileName, splitList):
        fileName = self.pathData + splitFileName
        lenList  = len(splitList)
        with open(fileName, 'w') as splitFile:
            for i in range(lenList):
                splitFile.write(splitList[i] + '.jpg')
                if (i == lenList - 1):
                    break
                splitFile.write('\n')
        return

    def loadImage(self, fileName, flagGray = False):
        try:
            return img_as_ubyte(imread(fileName, flagGray))
        
        except FileNotFoundError:
            print(' ERROR: Not found image file...')
            return np.zeros(1)

    def loadConfigFile(self):        
        try:
            with open(self.pathData + self.configFileName, 'r') as FileJSON:
                dataJSON = json.load(FileJSON)

        except FileNotFoundError:
            print(' ERROR: Not found config file...')
            return
        
        for key in self.CONFIGKEYS:
            self.configDict[key] = dataJSON[key]
                
        for label in dataJSON['labels']:
            self.labelDict[label['name']] = label['colors']
         
        colorArray = list(map(lambda elem: elem['colors'], dataJSON['grayLabels']))        
        grayLabelCount = 256 - len(colorArray)
        self.grayColor = [color for color in colorArray] + [[255, 255, 255]] * (grayLabelCount)
        
        return

    def loadJSONFileNames(self):
        files = listdir(self.pathData + self.configDict['dataLabels'])
        labelFiles = [file for file in files if file.endswith('.json')]
        return labelFiles

    def loadlabelImage(self, fileName):
        dataImage = None
        fileName  = self.pathData + self.configDict['dataLabels'] + fileName
        try:
            with open(fileName, 'r') as FileLabel:
                dataImage = json.load(FileLabel)

        except FileNotFoundError:
            print(' ERROR: Not found label file...')
        return dataImage

    def drawLines(self, coords, colorLeft, colorRight):
        minLen = min(len(coords[0]), len(coords[1]))
        for i in range(minLen):
            pointLeftRow  = coords[0][i][1]
            pointLeftCol  = coords[0][i][0]
            pointRightRow = coords[1][i][1]
            pointRightCol = coords[1][i][0]
            midPointRow = (pointLeftRow + pointRightRow) // 2
            midPointCol = (pointLeftCol + pointRightCol) // 2
            lineLeft  = draw.line(pointLeftRow, pointLeftCol,   midPointRow,   midPointCol)
            lineRight = draw.line( midPointRow,  midPointCol, pointRightRow, pointRightCol)
            draw.set_color(self.labelImage,  lineLeft,  colorLeft)
            draw.set_color(self.labelImage, lineRight, colorRight)
        return

    def drawPolygon(self, color, polygon, flagPer = True):
        polygonRow = [elem[1] for elem in polygon]
        polygonCol = [elem[0] for elem in polygon]
        if flagPer:
            polygonDraw = draw.polygon_perimeter(polygonRow, polygonCol)
        else:
            polygonDraw = draw.polygon(polygonRow, polygonCol)           
        draw.set_color(self.labelImage, polygonDraw, color)
        return

    def JSON2Image(self, JSONLabel):
        if JSONLabel is None:
            return None
        frameID    = JSONLabel['frame']       
        colorLeft  = self.labelDict['Switch-left']
        colorRight = self.labelDict['Switch-right']
        self.labelImage = np.zeros((JSONLabel['imgHeight'], JSONLabel['imgWidth'], 3), dtype = np.uint8)
        
        for objInImage in JSONLabel['objects']:      
            color = self.labelDict[objInImage['label'].capitalize()]
            
            if   'boundingbox'   in objInImage:
                boundingBox = objInImage['boundingbox'][::-1]
                rectangle = draw.rectangle(boundingBox[0:2], boundingBox[2:4])
                draw.set_color(self.labelImage, rectangle, color)
            
            elif 'polygon'       in objInImage:
                self.drawPolygon(color, objInImage['polygon'], False)
            
            elif 'polyline'      in objInImage:
                self.drawPolygon(color, objInImage['polyline'])
            
            elif 'polyline-pair' in objInImage:
                polyline2 = objInImage['polyline-pair']        
                self.drawLines(polyline2, colorLeft, colorRight)
                self.drawPolygon(color, polyline2[0])
                self.drawPolygon(color, polyline2[1])
        
        fileName = self.pathData + self.configDict['labelImages'] + frameID + '.jpg'
        self.saveImage(fileName, self.labelImage)      
        return frameID

    def grayRow2Color(self, row):      
        return [self.grayColor[elem] for elem in row]

    def gray2Color(self, frameID):
        fileName  = self.pathData + self.configDict['grayImages'] + frameID + '.png'
        imageGray = self.loadImage(fileName, True)
        grayImage = np.array(list(map(self.grayRow2Color, imageGray)), dtype = np.uint8)
        fileName = self.pathData + self.configDict['gray2Colors'] + frameID + '.jpg'
        self.saveImage(fileName, grayImage)
        return grayImage

    def blend3Image(self, frameID, grayImage):
        fileName   = self.pathData + self.configDict['dataImages']  + frameID + '.jpg'
        dataImage  = self.loadImage(fileName)
        blendImage = (      dataImage * self.configDict['blendConfig'][0] + \
                            grayImage * self.configDict['blendConfig'][1] + \
                      self.labelImage * self.configDict['blendConfig'][2]).astype(np.uint8)
        
        fileName = self.pathData + self.configDict['blendImages'] + frameID + '.jpg'
        self.saveImage(fileName, blendImage)
        return blendImage

    def processAllImage(self):
        JSONFileNameList = self.loadJSONFileNames()
        for labelFile in JSONFileNameList:
            start = thread_time()
            JSONLabel  = self.loadlabelImage(labelFile)
            frameID    = self.JSON2Image(JSONLabel)
            grayImage  = self.gray2Color(frameID)
            blendImage = self.blend3Image(frameID, grayImage)
            self.frameIDList.append(frameID)
            timeString = str(thread_time() - start)
            print(f'\t ImageID = {frameID}, time = {timeString} (sec)')
            #self.diplayImage(blendImage)
            #if frameID == 'rs00100':
            #    break
        return

    def splitDataImage(self):
        lenList = len(self.frameIDList)
        shuffle(self.frameIDList)
        offsetSplit = int(lenList * self.configDict['splitConfig'][0])
        self.saveSplitData(self.configDict['splitTrain'], self.frameIDList[:offsetSplit])
        self.saveSplitData(self.configDict['splitTest'],  self.frameIDList[offsetSplit:])
        print('\n\t Offset split data to train =', offsetSplit)
        return

    def main(self):
        print(' START...')
        start = thread_time()
        self.loadConfigFile()
        self.processAllImage()
        self.splitDataImage()
        timeString = str(thread_time() - start)
        print('\n\t Time for processing all data =', timeString, '(sec)')
        print('\n END...')
        return

if __name__ == '__main__':
    
    pathData       = './../Data/'
    configFileName = 'config.json'
    
    Label2Image(pathData, configFileName).main()