"""
Patrick Spieker 
Spring 2017 - HPCC Research
"""


from Tkinter import *
import xarray as xr
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers

ds = xr.open_dataset('./data/aggregated.nc')
nino34 = ds['sst'].sel(lat=slice(-6.0, 6.0), lon=slice(190, 240))
nino34 = nino34[:, 0, :, :]

demean = lambda df: df - df.mean()

#means = nino34.groupby('time.month').apply(np.mean)

nino34 = nino34.groupby('time.month') - nino34.groupby('time.month').apply(np.mean)
nino34 = nino34 / np.max([-1 * np.min(nino34.values), np.max(nino34.values)])
start_test = (nino34.shape[0] - 300)
x_train = nino34[:start_test, :, :].values.astype('float32')\
            .reshape(start_test, nino34.shape[1] * nino34.shape[2])
x_test = nino34[start_test:, :, :].values.astype('float32')\
            .reshape(nino34.shape[0] - start_test, \
            nino34.shape[1] * nino34.shape[2])

def train_dim(e_dim, x_train, x_test, dec_dict):
    dim = e_dim.get()
    # this is our input placeholder
    input_layer = Input(shape=(182,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(dim, activation='tanh')(input_layer)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(182, activation='tanh')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoded)

    rms_prop = optimizers.RMSprop(lr=0.0001)
    autoencoder.compile(optimizer=rms_prop, loss='mse')
    non_linear_history = autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=64,
                validation_data=(x_test, x_test),
                verbose=0)
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    encoded_input = Input(shape=(dim,))
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    dec_dict['dec'] = decoder
        
def gen_values(dec_dict, mtx, e_dim, dim_vals):
    dec = dec_dict['dec']
    ip = np.array([[float(l.get()) for l in dim_vals[0:int(e_dim.get())]]])
    output = dec.predict(ip)
    output = output.reshape((7,26))
    for i in xrange(1, 8):
        for j in xrange(1, 27):
            n = output[i-1][j-1]
            mtx[i-1][j-1]['text']=round(n, 2)

# setting up the main Tkinter framing stuff
root = Tk()
root.title("ENSO Simulator")

mainframe = Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# setting up the level of compression, 182 -> e_dim
Label(mainframe, text="Dim Comp.").grid(row=1, column=0)
e_dim = Scale(mainframe, from_=1, to_=6, orient=HORIZONTAL)
e_dim.set(3)
e_dim.grid(row=2, column=0)

# configuring the dim changing button
set_dim = Button(mainframe, text="Set dim",\
        command=lambda : train_dim(e_dim, x_train, x_test, dec_dict))
set_dim.grid(row=3, column=0)

# setting up the changable values widgets + labels
dim_vals = [Entry(mainframe) for i in range(6)]
for i,e in enumerate(dim_vals):
    e.insert(0, 1.0*i/8)
    Label(mainframe, text=str(i)).grid(row=4+2*i)
    e.grid(row=4+2*i+1, column=0)

# Setting up the actual matrix
x_offset = 1
y_offset = 0

# setting the decoder dictionary for passing around the decoder
dec_dict = {'dec': None}

# setting up the matrix of widgets we will pass around
mtx = [[] for i in range(8)]
for i in xrange(1, 8):
    for j in xrange(1, 27):
        l = Label(mainframe, text="%d %d" % (i, j), highlightthickness=3)
        l.grid(row=i+y_offset-1, column=j+x_offset-1)
        mtx[i-1].append(l)
#for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

# setting up the button to retrain the model
get_new_mtx = Button(mainframe, text="Generate values",\
        command=lambda : gen_values(dec_dict, mtx, e_dim, dim_vals))
get_new_mtx.grid(row=0, column=0)



root.bind('<Return>', lambda: regen(e_dim, x_train, x_test, dec_dict))

root.mainloop()
