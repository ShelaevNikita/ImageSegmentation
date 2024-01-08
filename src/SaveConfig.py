
import json

DATAPATH        = './../Data/'
CONFIGFILENAME  = 'config.json'

DATAIMAGEPATH   = 'dataImages/'
DATALABELPATH   = 'dataLabels/'
GRAYIMAGEPATH   = 'grayImages/'
LABELIMAGEPATH  = 'labelImages/'
GRAY2COLORIMAGE = 'gray2Colors/'
BLENDIMAGEPATH  = 'blendImages/'

SPLITFILETRAIN  = 'splitData/splitDataTrain.dat'
SPLITFILETEST   = 'splitData/splitDataTest.dat'

BLENDCONFIG     = (0.5, 0.35, 0.15)
SPLITCONFIG     = (0.8, 0.2)

LABEL2RGB = {
    'Buffer-stop'       :( 70,  70,  70),
    'Crossing'          :(128,  64, 128),
    'Guard-rail'        :(  0, 255,   0),
    'Train-car'         :(  0,  80, 100),
    'Platform'          :(244,  35, 232),
    'Rail'              :(  0, 255, 255),
    'Switch-indicator'  :(  0, 255, 127),
    'Switch-left'       :(255, 255,   0),
    'Switch-right'      :(  0, 127, 127),
    'Switch-unknown'    :(  0, 191, 191),
    'Switch-static'     :(127, 255, 255),
    'Track-sign-front'  :(220, 220,   0),
    'Track-signal-front':(250, 170,  30),
    'Track-signal-back' :(125,  85,   0),
    'Person-group'      :(220,  20,  60),
    'Car'               :(  0,   0, 142),
    'Fence'             :(190, 153, 153),
    'Person'            :(220,  20,  60),
    'Pole'              :(153, 153, 153),
    'Rail-occluder'     :(255, 255, 255),
    'Truck'             :(  0,   0,  70)
}

GRAYCOLOR = {
    'Road'         :(128,   0, 128),
    'Sidewalk'     :(244,  35, 232),
    'Construction' :( 70,  70,  70),
    'Tram-track'   :(192,   0, 128),
    'Fence'        :(190, 153, 153),
    'Pole'         :(153, 153, 153),
    'Traffic-light':(250, 170,  30),
    'Traffic-sign' :(220, 220,   0),
    'Vegetation'   :(107, 142,  35),
    'Terrain'      :(152, 251, 152),
    'Sky'          :( 70, 130, 180),
    'Human'        :(220,  20,  60),
    'Rail-track'   :(230, 150, 140),
    'Car'          :(  0,   0, 142),
    'Truck'        :(  0,   0,  70),
    'Trackbed'     :( 90,  40,  40),   
    'On-rails'     :(  0,  80, 100),
    'Rail-raised'  :(  0, 254, 254),
    'Rail-embedded':(  0,  68,  63)
}

def saveLabel2Config():
    labelArray     = [{'name':key, 'colors':value} for key, value in LABEL2RGB.items()]
    grayLabelArray = [{'name':key, 'colors':value} for key, value in GRAYCOLOR.items()]
    resultDict = {
        'dataImages' :  DATAIMAGEPATH,
        'dataLabels' :  DATALABELPATH,
        'grayImages' :  GRAYIMAGEPATH,
        'labelImages': LABELIMAGEPATH,
        'gray2Colors':GRAY2COLORIMAGE,
        'blendImages': BLENDIMAGEPATH,
        'splitTrain' : SPLITFILETRAIN,
        'splitTest'  :  SPLITFILETEST,
        'blendConfig':    BLENDCONFIG,
        'splitConfig':    SPLITCONFIG,
        'grayLabels' : grayLabelArray,
        'labels'     :     labelArray
    }
    with open(DATAPATH + CONFIGFILENAME, 'w') as FileJSON:
        json.dump(resultDict, FileJSON, indent = 4)   
    return

if __name__ == '__main__':
    saveLabel2Config()