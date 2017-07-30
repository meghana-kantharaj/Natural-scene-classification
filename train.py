#! /usr/bin/python
# -*- coding: utf-8 -*-
import ast
import pickle
import numpy as np
import os
import socket
import sys
import importlib
from datetime import datetime

import theano as th
import lib.neuralnet as nn

################################ HELPER FUNCTIONS ############################


def share(data, dtype=th.config.floatX, borrow=True):
    return th.shared(np.asarray(data, dtype), borrow=borrow)


def fixdim(arr):
    if arr.ndim == 2:
        side = int(arr.shape[-1] ** .5)
        assert side**2 == arr.shape[-1], "Need a perfect square"
        return arr.reshape((arr.shape[0], 1, side, side))

    if arr.ndim == 3:
        return np.expand_dims(arr, axis=1)

    if arr.ndim == 4:
        return arr

    raise ValueError("Image data arrays must have 2,3 or 4 dimensions only")


class WrapOut:
    def __init__(self, use_file, name=''):
        self.name = name
        self.use_file = use_file
        if use_file:
            self.stream = open(name, 'w', 1)
        else:
            self.stream = sys.stdout

    def write(self, data):
        self.stream.write(data)

    def forceflush(self):
        if self.use_file:
            self.stream.close()
            self.stream = open(self.name, 'a', 1)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

################################### MAIN CODE ################################

if len(sys.argv) < 3:
    print('Usage:', sys.argv[0],
          ''' <dataset> <params_file(s)> [redirect=0]
    dataset:
        Should be the name of a module in the data folder.
        Like "mnist", "telugu_ocr", "numbers" etc.
    params_file(s) :
        Parameters for the NeuralNet
        - name.prms : contains the initialization code
        - name.pkl  : pickled file from a previous run (has wts too).
    redirect:
        1 - redirect stdout to a params_<SEED>.txt file
    ''')
    sys.exit()
truePos=0
allOther=0
total=0
tpr=[]
fpr=[]
dataset_name = sys.argv[1]
prms_file_name = sys.argv[2]
matrix=[
[' ','coast','forest','mountain','tallbuilding'],
['coast',0,0,0,0],
['forest',0,0,0,0],
['mountain',0,0,0,0],
['tallBuilding',0,0,0,0]
]
##########################################  Import Parameters

if prms_file_name.endswith('.pkl'):
    with open(prms_file_name, 'rb') as f:
        params = pickle.load(f)
else:
    with open(prms_file_name, 'r') as f:
        params = ast.literal_eval(f.read())

layers = params['layers']
tr_prms = params['training_params']
try:
    allwts = params['allwts']
except KeyError:
    allwts = None

## Init SEED
if (not 'SEED' in tr_prms) or (tr_prms['SEED'] is None):
    tr_prms['SEED'] = np.random.randint(0, 1e6)

out_file_head = os.path.basename(prms_file_name,).replace(
    os.path.splitext(prms_file_name)[1], "_{:06d}".format(tr_prms['SEED']))

if sys.argv[-1] is '1':
    print("Printing output to {}.txt".format(out_file_head), file=sys.stderr)
    sys.stdout = WrapOut(True, out_file_head + '.txt')
else:
    sys.stdout = WrapOut(False)


##########################################  Print Parameters

print(' '.join(sys.argv), file=sys.stderr)
print(' '.join(sys.argv))
print('Time   :' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Device : {} ({})'.format(th.config.device, th.config.floatX))
print('Host   :', socket.gethostname())

print(nn.get_layers_info(layers))
print(nn.get_training_params_info(tr_prms))

##########################################  Load Data
data = importlib.import_module("data." + dataset_name)

tr_corpus_sz, n_maps, _, layers[0][1]['img_sz'] = data.training_x.shape
te_corpus_sz = data.testing_x.shape[0]
data.training_x = fixdim(data.training_x)
data.testing_x = fixdim(data.testing_x)

trin_x = share(data.training_x)
test_x = share(data.testing_x)
trin_y = share(data.training_y, 'int32')
test_y = share(data.testing_y, 'int32')

try:
    trin_aux = share(data.training_aux)
    test_aux = share(data.testing_aux)
except AttributeError:
    trin_aux, test_aux = None, None

print("\nInitializing the net ... ")
net = nn.NeuralNet(layers, tr_prms, allwts)
print('oooooooooooooooo000000000000000000000000000000000')
print(net)
print('oooooooooooooooo000000000000000000000000000000000')

print(net.get_wts_info(detailed=True).replace("\n\t", ""))
print('oooooooooooooooo000000000000000000000000000000000')

print("\nCompiling ... ")
training_fn = net.get_trin_model(trin_x, trin_y, trin_aux)
test_fn_tr = net.get_test_model(trin_x, trin_y, trin_aux)
test_fn_te = net.get_test_model(test_x, test_y, test_aux)

batch_sz = tr_prms['BATCH_SZ']
nEpochs = tr_prms['NUM_EPOCHS']
nTrBatches = tr_corpus_sz // batch_sz
nTeBatches = te_corpus_sz // batch_sz

############################################## MORE HELPERS 
def calc_eff(predicted,actual):
    for i in range(len(predicted)):
        if predicted[i]==actual[i]:
            truePosl=truePos+1
            global truePos 
            truePos=truePosl
            val= predicted[i]+1
            pos= matrix[val][val]
            global matrix
            matrix[val][val]=pos+1
        else:
            allOtherl=allOther+1
            global allOther
            allOther=allOtherl
            val = matrix[actual[i]+1][predicted[i]+1]
            global matrix
            matrix[actual[i]+1][predicted[i]+1]=val+1
        totall=total+1
        global total
        total =totall

    tp=[]
    if(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]!= 0):
        tp.append(matrix[1][1]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))

        tp.append(matrix[2][2]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
        tp.append(matrix[3][3]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
        tp.append(matrix[4][4]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
    else:
        tp.append(0);
    fp=[]
    if(matrix[1][3]+matrix[1][4]+matrix[1][2]+matrix[3][3]+matrix[4][4]+matrix[2][2]!=0):
        fp.append((matrix[1][3]+matrix[1][4]+matrix[1][2])/(matrix[1][3]+matrix[1][4]+matrix[1][2]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
    else:
        fp.append(0)
    if(matrix[2][1]+matrix[2][3]+matrix[2][4]+matrix[1][1]+matrix[3][3]+matrix[4][4]!=0):
        fp.append((matrix[2][1]+matrix[2][3]+matrix[2][4])/(matrix[2][1]+matrix[2][3]+matrix[2][4]+matrix[1][1]+matrix[3][3]+matrix[4][4]))
    else:
        fp.append(0)
    if(matrix[3][1]+matrix[3][2]+matrix[3][4]+matrix[1][1]+matrix[4][4]+matrix[2][2]!=0):
        fp.append((matrix[3][1]+matrix[3][2]+matrix[3][4])/(matrix[3][1]+matrix[3][2]+matrix[3][4]+matrix[1][1]+matrix[4][4]+matrix[2][2]))
    else:
        fp.append(0)
    if (matrix[4][1]+matrix[4][2]+matrix[4][3]+matrix[1][1]+matrix[3][3]+matrix[2][2]!=0):
        fp.append((matrix[4][1]+matrix[4][2]+matrix[4][3])/(matrix[4][1]+matrix[4][2]+matrix[4][3]+matrix[1][1]+matrix[3][3]+matrix[2][2]))
    else:
        fp.append(0)
    global tpr
    tpr.append(sum(tp)/len(tp))
    global fpr
    fpr.append(sum(fp)/len(fp))
    return

def test_wrapper(nylist):
    sym_err, bit_err, n = 0., 0., 0
    vals=[]
    for val in nylist:
        vals.append(val)

    for symdiff, bitdiff, preds, predicted, actual in vals:
        sym_err += symdiff
        bit_err += bitdiff
        n += 1
        calc_eff(predicted,actual)
    return 100 * sym_err / n, 100 * bit_err / n

if net.tr_layers[-1].kind == 'LOGIT':
    aux_err_name = 'BitErr'
else:
    aux_err_name = 'P(MLE)'


def get_test_indices(tot_samps, bth_samps=tr_prms['TEST_SAMP_SZ']):
    n_bths_each = int(bth_samps / batch_sz)
    n_bths_all = int(tot_samps / batch_sz)
    cur = 0
    while True:
        yield [i % n_bths_all for i in range(cur, cur + n_bths_each)]
        cur = (cur + n_bths_each) % n_bths_all


test_indices = get_test_indices(te_corpus_sz)
trin_indices = get_test_indices(tr_corpus_sz)
pickle_file_name = out_file_head + '_{:02.0f}.pkl'
saved_file_name = None


def do_test():
    global saved_file_name
    test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                          for i in next(test_indices))
    trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                          for i in next(trin_indices))
    print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
    sys.stdout.forceflush()

    if saved_file_name:
        os.remove(saved_file_name)

    saved_file_name = pickle_file_name.format(test_err)
    with open(saved_file_name, 'wb') as pkl_file:
        pickle.dump(net.get_init_params(), pkl_file, -1)

############################################ Training Loop

print("Training ...")
print("Epoch   Cost  Tr_Error Tr_{0}    Te_Error Te_{0}".format(aux_err_name))
for epoch in range(nEpochs):
    total_cost = 0

    for ibatch in range(nTrBatches):
        output = training_fn(ibatch)
        total_cost += output[0]
        if(ibatch>10 and ibatch<20):
            print('cost ',ibatch,' : ',total_cost)
        if np.isnan(total_cost):
            print("Epoch:{} Iteration:{}".format(epoch, ibatch))
            print(net.get_wts_info(detailed=True))
            raise ZeroDivisionError("Nan cost at Epoch:{} Iteration:{}"
                                    "".format(epoch, ibatch))

    if epoch % tr_prms['EPOCHS_TO_TEST'] == 0:
        print("{:3d} {:>8.2f}".format(net.get_epoch(), total_cost), end='    ')
        do_test()
        if  total_cost > 1e6:
            print(net.get_wts_info(detailed=True))

    net.inc_epoch_set_rate()

########################################## Final Error Rates

test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                      for i in range(te_corpus_sz//batch_sz))
trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                      for i in range(tr_corpus_sz//batch_sz))

print("{:3d} {:>8.2f}".format(net.get_epoch(), 0), end='    ')
print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
print('------------------------total--------------------------')
print('total :  ',total)
print('truePos  : ',truePos)
print('efficiency : ',truePos/total)
print('-------------------------------------------------------');
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(fpr,tpr , 'x')



fig.suptitle('Roc curve', fontsize=14)
plt.xlabel('false positive rate', fontsize=14)
plt.ylabel('true positive rate', fontsize=14)
fig.savefig('roc.jpg')
plt.show()


import tkinter as tk

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
    frame3=tk.Frame(frame,width=300, height=300,pady=20,padx=20, bg="#848484" , colormap="new")
    frame3.pack()
    path = "roc.jpg"
    im = Image.open(path)
    tkimage = ImageTk.PhotoImage(im)
    lb=tk.Label(frame3, image=tkimage)
    lb.image=tkimage
    lb.pack()
def showconfusion(frame,matrix):
    clear()
    global frame2
    frame2=tk.Frame(frame,width=300, height=300,pady=20,padx=20, bg="#848484" , colormap="new")
    frame2.pack()
    confusion = tk.Label(frame2, text="Confusion Matrix", fg="#111111", bg="#848484",font=("Helvetica", 20),pady=5)
    confusion.pack()
    t = SimpleTable(frame2, 5,5)
    t.pack(side="top", fill="x")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            t.set(i,j,matrix[i][j])

window = tk.Tk()
window.configure(background="#848484")
window.title("Scene Labeling")
w = 1000 # width for the Tk root
h = 800 # height for the Tk root
ws = window.winfo_screenwidth() # width of the screen
hs = window.winfo_screenheight() # height of the screen
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
window.geometry('%dx%d+%d+%d' % (w, h, x, y))

lblInst = tk.Label(window, pady=50,padx=20, wraplength=700, text="Natural scene classification using convolutional neural network", fg="#111111", bg="#848484", font=("Helvetica", 24))
lblInst.pack()
lbl2 = tk.Label(window, pady=50,padx=5,text="Overall efficiency : "+str((truePos/total)*100)+"%", fg="#111111", bg="#848484", font=("Helvetica", 18))
lbl2.pack()
framebuttons = tk.Frame(width=768, height=576, bg="", colormap="new")
framebuttons.pack()

frame = tk.Frame(window,width=500, height=500,pady=20,padx=20, bg="#848484" , colormap="new")
frame.pack()
B = tk.Button(framebuttons, text ="Show confusion", command = lambda: showconfusion(frame,matrix) )
B.pack(side='left')
B2 = tk.Button(framebuttons, text ="Show Roc", command = lambda: showroc(frame) )
B2.pack(side='left')
B3 = tk.Button(framebuttons, text ="clear", command = lambda: clear() )
B3.pack(side='left')
window.mainloop()