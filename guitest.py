import tkinter as tk
import time
from PIL import ImageTk, Image
class SimpleTable(tk.Frame):
    def __init__(self, parent, rows=10, columns=2):

        tk.Frame.__init__(self, parent, background="black")
        self._widgets = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                label = tk.Label(self, text="%s/%s" % (row, column), 
                                 borderwidth=0, width=10,pady=10,)
                label.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
                current_row.append(label)
            self._widgets.append(current_row)

        for column in range(columns):
            self.grid_columnconfigure(column, weight=1)
    
    def set(self, row, column, value):
        widget = self._widgets[row][column]
        widget.configure(text=value)

def showconfusion(frame,matrix):
    clear()
    global frame2
    frame2=tk.Frame(frame,width=300, height=300,pady=20,padx=20, bg="blue" , colormap="new")
    frame2.pack()
    confusion = tk.Label(frame2, text="Confusion Matrix", fg="#111111", bg="#848484",font=("Helvetica", 20),pady=5)
    confusion.pack()
    t = SimpleTable(frame2, 5,5)
    t.pack(side="top", fill="x")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            t.set(i,j,matrix[i][j])
def clear():
        try:
            frame2.destroy()
            
        except UnboundLocalError:
            print('handled UnboundLocalError')
        except NameError:
            print('handled NameError')
        try:
            frame3.destroy()
            
        except UnboundLocalError:
            print('handled UnboundLocalError')
        except NameError:
            print('handled NameError')
def showroc(frame):
    clear()
    global frame3
    frame3=tk.Frame(frame,width=300, height=300,pady=20,padx=20, bg="yellow" , colormap="new")
    frame3.pack()
    path = "arnat59.jpg"
    im = Image.open(path)
    tkimage = ImageTk.PhotoImage(im)
    lb=tk.Label(frame3, image=tkimage)
    lb.image=tkimage
    lb.pack() 


window = tk.Tk()
window.configure(background="#848484")
window.title("Scene Labeling")
w = 1000 # width for the Tk root
h = 700 # height for the Tk root
ws = window.winfo_screenwidth() # width of the screen
hs = window.winfo_screenheight() # height of the screen
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
window.geometry('%dx%d+%d+%d' % (w, h, x, y))

lblInst = tk.Label(window, pady=20,padx=20, wraplength=1000, text="Natural scene classification using convolutional neural network", fg="#111111", bg="#848484", font=("Helvetica", 24))
lblInst.pack()

framebuttons = tk.Frame(window,width=200, height=100,pady=20,padx=20, bg="#111111", colormap="new")
framebuttons.pack()
#frameWrap=tk.Frame(width=200, height=100,pady=20,padx=20, bg="green", colormap="new")
#frameWrap.pack()
matrix=[
[' ','coast','forest','mountain','tallbuilding'],
['coast',21,21,32,54],
['forest',654,12,656,32],
['mountain',456,456,345,3],
['tallBuilding',123,24,3,6]
]
frame = tk.Frame(window,width=500, height=500,pady=20,padx=20, bg="pink" , colormap="new")
frame.pack()
B = tk.Button(framebuttons, text ="Show confusion", command = lambda: showconfusion(frame,matrix) )
B.pack(side='left')
B2 = tk.Button(framebuttons, text ="Show Roc", command = lambda: showroc(frame) )
B2.pack(side='left')
B3 = tk.Button(framebuttons, text ="clear", command = lambda: clear() )
B3.pack(side='left')
window.mainloop()
