from tkinter import Frame, Tk, BOTH, Text, Menu, END, Label, TOP, BOTTOM
from tkinter import filedialog
from tkinter.constants import LEFT
from features import segments, to_mfcc, get_audio, segment_one, remove_silence, get_wiggles
from accuracy import predict_class_audio, get_lang
from tensorflow.keras.models import load_model

# Global vars no more pls
    
root = Tk()
redFrame = Frame(root, bg='lightblue')
blueFrame = Frame(root, bg='lightgreen')

def model_predict(audio):
    filtered = remove_silence(audio)
    mfcc = to_mfcc(filtered)
    segmented = segment_one(mfcc)
    model = load_model('model1.h5')
    return mfcc, predict_class_audio(segmented, model)

def processAudio():
    # print("i get to here")
    ftypes = [('Audio files', '*.mp3'), ('All files', '*')]
    dlg = filedialog.Open(root, filetypes = ftypes)
    audioFile = dlg.show()

    inputName = audioFile.split("/")[-1]
    languageInput = ''.join([i for i in inputName if not i.isdigit()]).split('.')[0]
    audio = get_audio(f'./data/{languageInput}/{inputName}')
    
    mfcc, prediction = model_predict(audio)
    image = get_wiggles(mfcc, blueFrame)
    image.get_tk_widget().pack(side=LEFT)
    # print()
     #Creating a Label to display text
    myLabel = Label(redFrame, text=inputName)
    myLabel.pack(side= LEFT)



    # Extract features and pass it to the model

    answer = prediction
    lang, answer = get_lang(languageInput, answer)
    #Creating a Label to display text
    languageLabel = Label(blueFrame, text=lang)
    myLabel = Label(blueFrame, text=answer)
    languageLabel.pack(side=BOTTOM)
    myLabel.pack(side= BOTTOM)



def main():
    root.wm_title('NLP Project')
    # Input Frame
    redFrame.pack_propagate(0)
    redFrame.pack(fill='both', side='top', expand='True')
     #Creating a Label to display text
    myLabel = Label(redFrame, text="Hello NLPeers ")
    myLabel.pack(side= TOP)

    #Creating a Label to display text
    myLabel = Label(redFrame, text="Select Audio file: ")
    myLabel.pack(side="left", padx=20, pady=20)

    
    

    #Output Frame
    blueFrame.pack_propagate(0)
    blueFrame.pack(fill='both', side='bottom', expand='True')



    # Select audio file
    menubar = Menu(root)
    root.config(menu=menubar)

    fileMenu = Menu(menubar)
    fileMenu.add_command(label="Open", command=lambda: processAudio())
    menubar.add_cascade(label="File", menu=fileMenu)  

    
    root.geometry("750x400")
    root.mainloop()  


if __name__ == '__main__':
    main() 