from pandas.io.pytables import attribute_conflict_doc
from pydub import AudioSegment
import os
import sys
import pandas as pd

class AudioSet:

    def __init__(self, csvPath, folder='./data') -> None:
        self.csvPath = csvPath
        self.folder = folder
        try:
            self.audioDF = pd.read_csv(csvPath)
        except:
            self.audioDF = pd.DataFrame(columns=['Audio', 'features', 'subject','spanish','english','chinese'])
            

    def check_folder(self):
        if not os.path.exists(self.folder):
            raise Warning('This folder does not exists')
        else:
            print('All good')


    def set_up(self):
        self.check_folder
        for lang in ['spanish', 'english']:
            cc = 0
            for i in range(500):
                if i% 250 == 0:
                    print('Halfway there')
                filename =f'./data/{lang}/' +lang + str(i) + '.mp3'
                try:
                    file = AudioSegment.from_mp3(filename)
                    file.export(f'./data/{lang}/{lang + str(cc)}.wav', format='wav')
                except:
                    print(f'Nothing in {lang} {i}')
                    cc += 1
                    continue
                cc += 1
                
    def data_setup(self):
        self.check_folder
        for lang in ['spanish', 'english']:
            cc = 0
            for i in range(500):
                filename =f'./data/{lang}/' +lang + str(i) + '.wav'
                try:
                    file = AudioSegment.from_mp3(filename)
                except:
                    continue
                features = None
                sub = lang + str(cc)
                spanish = 1 if lang == 'spanish' else 0
                english = 1 if lang == 'english' else 0
                chinese = 0
                
                self.audioDF.loc[len(self.audioDF.index)] = [filename, features, sub, spanish, english, chinese]
                # print(self.audioDF)
                cc+=1
                # break
            # break
        # print('l')
        self.audioDF.to_csv('./data.csv')
    


if __name__ == '__main__':
    newD = AudioSet('./data.csv')
    newD.data_setup()