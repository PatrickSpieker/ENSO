"""
Patrick Spieker 
Spring 2017 - HPCC Research
"""


from Tkinter import *
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as pyplot
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
set_dim = Button(mainframe, text="Train on dimension",\
        command=lambda : train_dim(e_dim, x_train, x_test, dec_dict))
set_dim.grid(row=3, column=0)


# setting up the changable values widgets + labels
Label(mainframe, text="Dimensional Values").grid(row=4, column=0)

dim_vals = [Scale(mainframe, from_=-1, to_=1,
                  orient=HORIZONTAL, resolution=0.1) for i in range(6)]
for i,e in enumerate(dim_vals):
    e.set(0)
    Label(mainframe, text=str(i+1)).grid(row=5+2*i)
    e.grid(row=5+2*i+1, column=0)

# Setting up the actual matrix
x_offset = 1
y_offset = 0

# setting the decoder dictionary for passing around the decoder
dec_dict = {'dec': None}

def draw_figure(master, fig_canvas, fig, loc=(0,0)):
    fig_canvas.draw()
    """
    fig_x, fig_y, fig_w, fig_h = fig.bbox.bounds
    fig_w, fig_h = int(fig_w), int(fig_h)
    photo = PhotoImage(master=master, width=fig_w, height=fig_h)

    canvas.create_image(loc[0] + fig_w/2, loc[1] + fig_h/2, image=photo)
    tkagg.blit(photo, fig_canvas.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo
    """


# generating the values for the new matrix
def gen_values(fig_canvas, quad, dec_dict, e_dim, dim_vals):
    dec = dec_dict['dec']
    ip = np.array([[float(l.get()) for l in dim_vals[0:int(e_dim.get())]]])
    output = dec.predict(ip)
    output = output.reshape((7*26,))
    print(output)
    quad.set_array(output)
    fig_canvas.draw()
    

# start out the viz with just zeroes on the heatmap
init_values = np.zeros(shape=(7,26))
f = Figure(figsize=(7,4))
a = f.add_subplot(111)
quad = a.pcolormesh(init_values, vmin=-0.5, vmax=0.5, cmap='RdBu')
fig_x, fig_y = 100, 100


fig_canvas  = FigureCanvasTkAgg(f, master=mainframe )
fig_canvas.get_tk_widget().\
        grid(row = y_offset, column = x_offset, rowspan=20, columnspan=10)
fig_canvas.draw()
fig_canvas.show()

# setting up the button to retrain the model
get_new_mtx = Button(mainframe, text="Generate values",\
        command=lambda : gen_values(fig_canvas, 
            quad, dec_dict, e_dim, dim_vals))
get_new_mtx.grid(row=0, column=0)

root.mainloop()
