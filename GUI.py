import tkinter
from PIL import ImageTk
import PIL.Image
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

def get_data(event):
    outlabel = Label(frame1,width=500,bg="white",font="Helvetica 12 bold")
    
    file=strVar.get()
    print("input:", strVar.get())

    model=['SVM(RBF)','SVM(L)','SVM','NB','KNN']
    #call our functions to return array
    total_data=['Angry ','Sad','Angry','Sad','Angry']
    
    data_from_all='\nSTATISTICS:\n---------------------'
    for i in range(len(total_data)):
        data_from_all=data_from_all+'\n'+model[i]+' : '+total_data[i]

    
    output='Surprise' #from voting func
    out_data='\nFinal Emotion : '+output

    if output == 'Angry':
        image_path="angry.png"
        img=PIL.Image.open(image_path)
    if output == 'Sad':
        image_path="sad.png"
        img=PIL.Image.open(image_path)
    if output == 'Happy':
        image_path="happy.png"
        img=PIL.Image.open(image_path)
    if output == 'Fear':
        image_path="fear.png"
        img=PIL.Image.open(image_path)
    if output == 'Surprise':
        image_path="surprise.png"
        img=PIL.Image.open(image_path)
        

    render=ImageTk.PhotoImage(img)
    panel=Label(frame1,image=render)
    panel.image=render
    
    outlabel.config(text=data_from_all+out_data)    
    outlabel.pack()
    panel.place(x=350,y=130)

def browsefunc():
    filename = filedialog.askopenfilename()
    strVar.set(filename)

root = Tk()
root.title('PROJECT')
root.configure(background='white',highlightthickness=2)
root.geometry("500x220+230+230")

frame = Frame(root, bg="white", highlightbackground="#111")
frame1=Frame(root, bg="white", highlightbackground="#111")
frame.pack()
frame1.pack(side = BOTTOM)


strVar = StringVar()
strVar.set("choose file")

strEntry = Entry(frame, textvariable=strVar,width=80,bg="#D3D3D3")
strEntry.pack()

browsebutton = Button(frame, text="Browse", command=browsefunc,bg="white")4
browsebutton.pack(side=RIGHT)


getDataButton = Button(frame, text="Get Data",bg="white")
getDataButton.bind("<Button-1>", get_data)
getDataButton.pack(side=RIGHT)

root.mainloop()
