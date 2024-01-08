
from src import UNETModel

def main():
    
    pathData       = './../Data/'
    configFileName = 'config.json'

    UNETModel.UNETModel(pathData, configFileName, 1).main()
    return

if __name__ == '__main__':
    main()