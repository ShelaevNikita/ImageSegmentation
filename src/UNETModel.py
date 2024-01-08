
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import Model, layers, utils
# from keras import losses

from time import thread_time

class UNETModel():
    
    CONFIGKEYS = ['dataImages', 'blendImages', 'splitTrain', 'splitTest']

    BatchSize  = 32
    ImageShape = (512, 512)
    
    Epochs = 10
    
    fileNameWeights = 'modelWeights.keras'

    def __init__(self, pathData, configFileName, repeat):
        self.pathData       = pathData
        self.configFileName = configFileName
        self.repeat         = repeat

        self.configDict = {}
        
        self.trainSize = 0
        self.imageNameList = []        

    def loadConfigFile(self):        
        try:
            with open(self.pathData + self.configFileName, 'r') as FileJSON:
                dataJSON = json.load(FileJSON)

        except FileNotFoundError:
            print(' ERROR: Not found config file...')
            return
        
        for key in self.CONFIGKEYS:
            self.configDict[key] = dataJSON[key]
            
        return

    def displayImage(self, displayList):
        plt.figure(figsize = (12, 8))
        title = ['Image', 'True Mask', 'Predicted Mask']
        for i in range(len(displayList)):
            plt.subplot(1, len(displayList), i + 1)
            plt.title(title[i])
            plt.imshow(utils.array_to_img(displayList[i]))
        plt.show()
        return

    def displayMetrics(self, history):
        accuracy = history['accuracy']
        loss     = history['loss']
        valAcc   = history['val_accuracy']
        valLoss  = history['val_loss']
        x        = [i for i in range(1, len(accuracy) + 1)]        

        _, axs = plt.subplots(ncols = 2, figsize = (12, 8), layout = 'constrained')
        
        axs[0].plot(x, loss,    label = 'Train data')
        axs[0].plot(x, valLoss, label = 'Test data')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss function')
        axs[0].set_title('Loss for epoch')
        axs[0].legend()
        
        axs[1].plot(x, accuracy, label = 'Train data')
        axs[1].plot(x, valAcc,   label = 'Test data')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy for epoch')
        axs[1].legend()
     
        plt.show()
        return

    def loadSplitData(self, splitFileName):
        fileName  = self.pathData + splitFileName
        try:
            with open(fileName, 'r') as splitFile:
                for line in splitFile.readlines():
                    self.imageNameList.append(line.replace('\n', ''))
                    
        except FileNotFoundError:
            print(' ERROR: Not found split file...')
        
        return

    def loadImageNames(self):
        self.loadSplitData(self.configDict['splitTrain'])
        self.trainSize = len(self.imageNameList)
        self.loadSplitData(self.configDict['splitTest'])
        return

    def loadImages(self, imageName):
        dataImageFile = self.pathData + self.configDict['dataImages']  + imageName
        maskImageFile = self.pathData + self.configDict['blendImages'] + imageName
        
        image = tf.io.read_file(dataImageFile)
        image = tf.io.decode_jpeg(image)
        image = tf.image.resize(image, self.ImageShape, method = 'nearest')
        image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
        
        mask  = tf.io.read_file(maskImageFile)
        mask  = tf.io.decode_jpeg(mask)
        mask  = tf.image.resize(mask, self.ImageShape, method = 'nearest')
        mask  = tf.image.convert_image_dtype(mask, tf.float32)
        mask  = tf.reshape(mask, [self.ImageShape[0], self.ImageShape[1], 3])

        return (image, mask)

    def augImage(self, image, mask):
        randomCrop = tf.random.uniform((), 0.5, 1)
        image = tf.image.central_crop(image, randomCrop)
        mask  = tf.image.central_crop(mask, randomCrop)
        
        randomFlip = tf.random.uniform((), 0, 1)
        if randomFlip >= 0.5 and randomFlip < 0.9:
            image = tf.image.flip_left_right(image)
            mask  = tf.image.flip_left_right(mask)
        
        if randomFlip >= 0.9:
            image = tf.image.flip_up_down(image)
            mask  = tf.image.flip_up_down(mask)

        image = tf.image.resize(image, self.ImageShape)
        mask  = tf.image.resize(mask, self.ImageShape)
        return (image, mask)
    
    '''
    def DICECoef(self, yPred, yTrue):
        yAPred = tf.unstack(yPred, axis = 3)
        yATrue = tf.unstack(yTrue, axis = 3)
        
        diceSum = 0
        for (YPred, YTrue) in zip(yAPred, yATrue):
            diceSum += (2 * tf.math.reduce_sum(YPred * YTrue) + 1) / (tf.math.reduce_sum(YPred + YTrue) + 1)
            
        return diceSum / self.classes
    
    def DICELoss(self, yPred, yTrue):
        return 1 - self.DICECoef(yPred, yTrue)

    def funLoss(self, yPred, yTrue):
        return 0.3 * self.DICELoss(yPred, yTrue) + losses.mean_squared_error(yTrue, yPred)
    '''
    
    def downSampleBlock(self, sample, sampleFilter):
        sampleA = layers.Conv2D(sampleFilter, 3, activation = 'relu', strides = 2,
                                padding = 'same', kernel_initializer = 'glorot_normal')(sample)
        
        sampleA = layers.BatchNormalization()(sampleA)
        sampleB = layers.Dropout(0.1)(sampleA)
        return (sampleA, sampleB)

    def upSampleBlock(self, sample, sampleFilter, sampleConc):
        sampleNew = layers.Conv2DTranspose(sampleFilter, 3, activation = 'relu', strides = 2,
                                           padding = 'same', kernel_initializer = 'glorot_normal')(sample)
        
        sampleNew = layers.Concatenate()([sampleNew, sampleConc])
        sampleNew = layers.Dropout(0.1)(sampleNew)
        return sampleNew

    def UNETModelFun(self):
        # Input
        inputLayer = layers.Input(shape = self.ImageShape + (3,))
        
        # Encoder
        f1, p1 = self.downSampleBlock(inputLayer, 32)
        f2, p2 = self.downSampleBlock(p1,  64)
        f3, p3 = self.downSampleBlock(p2, 128)
        f4, p4 = self.downSampleBlock(p3, 256)
        f5, p5 = self.downSampleBlock(p4, 512)
        
        # Bottleneck
        bottleNeck = layers.Conv2D(512, 3, activation = 'relu', strides = 2,
                                   padding = 'same', kernel_initializer = 'glorot_normal')(p5)
        
        # Decoder
        u1 = self.upSampleBlock(bottleNeck, 512, f5)
        u2 = self.upSampleBlock(u1, 256, f4)
        u3 = self.upSampleBlock(u2, 128, f3)
        u4 = self.upSampleBlock(u3,  64, f2)
        u5 = self.upSampleBlock(u4,  32, f1)
        
        # Output
        outputLayer = layers.Conv2DTranspose(3, 3, activation = 'sigmoid', strides = 2,
                                             padding = 'same', kernel_initializer = 'glorot_normal')(u5)
        
        model = Model(inputs = inputLayer, outputs = outputLayer, name = 'U-Net')
        return model

    def startModel(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.imageNameList)
        dataset = dataset.map(self.loadImages, num_parallel_calls = tf.data.AUTOTUNE)
        
        dataset = dataset.repeat(self.repeat)
        dataset = dataset.map(self.augImage, num_parallel_calls = tf.data.AUTOTUNE)
        
        trainSize = self.trainSize * self.repeat
        trainDataset = dataset.take(trainSize).cache()
        testDatasetA = dataset.skip(trainSize).take(len(dataset) - trainSize).cache()
        trainDataset = trainDataset.batch(self.BatchSize)
        testDataset  = testDatasetA.batch(self.BatchSize)
        
        imageSegModel = self.UNETModelFun()
        
        imageSegModel.compile(optimizer = 'adam', loss = 'mean_squared_error',
                              metrics = ['accuracy'])
        
        imageSegModel.summary()
        
        modelHistory = imageSegModel.fit(trainDataset, validation_data = testDataset, epochs = self.Epochs)
        
        imageSegModel.save(self.pathData + self.fileNameWeights)
        self.displayMetrics(modelHistory.history)
        return (imageSegModel, testDatasetA)

    def showPrediction(self, model, dataset, skip):
        model.load_weights(self.pathData + self.fileNameWeights)
        
        data  = list(dataset.skip(skip).take(1).as_numpy_iterator())
        image = data[0][0]
        mask  = data[0][1]

        imageModel = tf.reshape(image, [1, self.ImageShape[0], self.ImageShape[1], 3])
        predMask = model.predict(imageModel)
        predMask = tf.reshape(predMask, [self.ImageShape[0], self.ImageShape[1], 3])
        
        self.displayImage([image, mask, predMask])
        return

    def main(self):
        print(' START...')
        start = thread_time()
        self.loadConfigFile()
        self.loadImageNames()
        imageSegModel, testDataset = self.startModel()
        self.showPrediction(imageSegModel, testDataset, 100)
        timeString = str(thread_time() - start)
        print('\n\t Time for processing all data =', timeString, '(sec)')
        print('\n END...')
        return

if __name__ == '__main__':
    
    pathData       = './../Data/'
    configFileName = 'config.json'

    UNETModel(pathData, configFileName, 1).main()
