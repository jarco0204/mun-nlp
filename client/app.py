from tkinter import Frame, Tk, BOTH, Text, Menu, END, Label, TOP, BOTTOM
from tkinter import filedialog
from tkinter.constants import LEFT


# Global vars 
root = Tk()
redFrame = Frame(root, bg='lightblue')
blueFrame = Frame(root, bg='lightgreen')


       

       
def processAudio():
    print("i get to here")
    ftypes = [('Audio files', '*.mp3'), ('All files', '*')]
    dlg = filedialog.Open(root, filetypes = ftypes)
    audioFile = dlg.show()

    inputName = audioFile.split("/")[-1]
    # print()
     #Creating a Label to display text
    myLabel = Label(redFrame, text=inputName)
    myLabel.pack(side= LEFT)



    # Extract features and pass it to the model

    answer = "ELPEPE"

    #Creating a Label to display text
    myLabel = Label(blueFrame, text=answer)
    myLabel.pack(side= BOTTOM)



def main():
    

    # Input Frame
    redFrame.pack_propagate(0)
    redFrame.pack(fill='both', side='top', expand='True')
     #Creating a Label to display text
    myLabel = Label(redFrame, text="Hello NLPeers ")
    myLabel.pack(side= TOP)

    #Creating a Label to display text
    myLabel = Label(redFrame, text="Select Audio file to predict label with a 2D Convolutional NN: ")
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

    
    root.geometry("500x250+300+300")
    root.mainloop()  


if __name__ == '__main__':
    main() 