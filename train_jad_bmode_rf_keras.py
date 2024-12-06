import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from tensorflow_addons.callbacks import TQDMProgressBar
import neurite as ne
from tqdm import tqdm
import glob
from scipy.io import loadmat

# import voxelmorph with pytorch backend
# os.environ['NEURITE_BACKEND'] = 'pytorch'
# os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import bmode_rf_network
import generators
import losses


def plot_history(hist):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history['loss'], '.-', label='Training Loss')
    plt.plot(hist.epoch, hist.history['val_loss'], '.-', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('VoxelMorph Training Loss')
    
    plt.savefig(prefix + '_loss.png')

# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------


folders_path = 'Jad_RFData'                     # input image directory
prefix = 'Bmode_RF_MSE'                        # output model name prefix
gpus = [0]
device = 'cuda:0'
cudnn_nondet = True                             # disable cudnn determinism - might slow down training
bidirectional = False                           # enable bidirectional cost function (not implemented)
batch_size = 4
lr = 1e-4                                       # learning rate (default: 1e-4)
epochs = 50                                      # number of training epochs (default: 1500)
steps_per_epoch = 150                           # number of training batches per epoch (default: 100)
val_steps_per_epoch = 30
initial_epoch = 0                               # initial epoch number (default: 0)
debug = False                                   # when debug = True, script only loads two scans and trains for two epochs
ncc = False


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
wd=176 #3*128
ht=1312 #128*6 
fixed_bmode = []
moving_bmode = []
fixed_rf = []
moving_rf = []

folders = glob.glob(folders_path + '/*')

print("Loading B-mode images and RF data...")
with tqdm(total=90) as pbar:
    for i, folder_path in enumerate(folders):
        sims = glob.glob(folder_path + '/*')

        for j, sim_path in enumerate(sims):
            skip = False

            bmode_path = Path(glob.glob(sim_path+'/ppb.mat')[0])
            bmode = loadmat(bmode_path)['BmodeRFScanConv']
            num_slices = bmode.shape[2]

            rf_path = Path(glob.glob(sim_path+'/ConvRF.mat')[0])
            rf = loadmat(rf_path)['rfScanConv']
            num_rf_slices = rf.shape[2]
            #assert num_rf_slices == num_slices, 'Numbers of B-mode/RF frames should be the same. B-mode frames: %s, Rf frames: %s' % (num_slices, num_rf_slices)
            if num_rf_slices != num_slices: 
                print("wrong number of frames found, skipping scan")
                continue

            for slice_num in range(num_slices):
                slice = bmode[:,:,slice_num]
                slice = np.nan_to_num(slice, nan=np.nanmax(slice)) # changing nans to lowest value
                slice = cv2.resize(slice, (wd,ht), interpolation=cv2.INTER_NEAREST)
                slice = slice / np.max(np.absolute(slice))

                rf_slice = rf[:,:,slice_num]
                rf_slice = np.nan_to_num(rf_slice, nan=np.nanmax(rf_slice)) # changing nans to lowest value # changing nans to 0  

                rf_slice = np.log1p(np.abs(rf_slice-0.5))
                rf_slice = cv2.resize(rf_slice, (wd,ht), interpolation=cv2.INTER_NEAREST)
                rf_slice = rf_slice / np.max(np.absolute(rf_slice))
                rf_slice = -1 + (rf_slice + 1)/2

                # This block loads consecutive pairs of images:
                if slice_num > 0: 
                    fixed_bmode.append(slice)
                    fixed_rf.append(rf_slice)
                if slice_num < num_slices-1:
                    moving_bmode.append(slice)
                    moving_rf.append(rf_slice)  
                       
            pbar.update()

            if j == 0 and debug: break
        if i == 2 and debug == True: break
pbar.close()

fixed_bmode = np.array(fixed_bmode)
moving_bmode = np.array(moving_bmode)
fixed_rf = np.array(fixed_rf)
moving_rf = np.array(moving_rf)

assert fixed_bmode.shape == fixed_rf.shape, 'Dimensions and numbers of B-mode/RF frames should be the same. B-mode dims: %s, Rf dims: %s' % (fixed_bmode.shape, fixed_rf.shape)

fixed = np.stack((fixed_bmode, fixed_rf), axis=3) # (slices, wd, ht, 2) <- bmode = 0, rf = 1
moving = np.stack((moving_bmode, moving_rf), axis=3) # (slices, wd, ht, 2)

train_fixed, val_fixed, train_moving, val_moving = train_test_split(moving, fixed, test_size=0.2, random_state=50)

print("Training Dataset Length: %d" % len(train_fixed))
print("Validation Dataset Length: %d" % len(val_fixed))

train_generator = generators.generator_4D(train_moving[...,0], train_fixed[..., 0], train_moving[..., 1], train_fixed[..., 1], batch_size=batch_size)
val_generator = generators.generator_4D(val_moving[...,0], val_fixed[..., 0], val_moving[..., 1], val_fixed[..., 1], batch_size=batch_size)


# Uncomment these blocks to visualize inputs
# for i in range(5):
#     in_sample, out_sample = next(train_generator)
#     images = [cv2.resize(img[0, :, :, 0], (512, 512), interpolation=cv2.INTER_NEAREST) for img in in_sample + out_sample] 
#     titles = ['moving', 'fixed', 'moving rf', 'fixed rf', 'moved ground-truth (fixed)', 'zeros']
#     ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
#     plt.savefig(prefix + 'testdata.png')

# # in_sample, out_sample = next(train_generator)
# # plt.imshow(in_sample[0][3, :, :, 1, 0])
# # plt.savefig("test.png")

# exit()

# ----------------------- MODEL CREATION -----------------------

# configure unet features 
nb_features = [
    [16, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 32, 16]  # decoder features
]

# build model using custom model Vxm4D
inshape = moving_bmode.shape[1:]
vxm_model = bmode_rf_network.Vxm4D(inshape, nb_features, int_steps=0)

# instantiate losses
if ncc:
    loss_weights = [-1, 0.01]   
    losses = [vxm.losses.NCC(win=[10, 45]).loss, vxm.losses.Grad('l2').loss]
else:
    loss_weights = [100, 5]
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
    factor=0.1,
    patience=3,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.0
)

early_stop = EarlyStopping(monitor='val_loss',
    min_delta=0.00001,
    patience=15,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

# ----------------------- MODEL TRAINING -----------------------

# val_steps_per_epoch = len(moving_bmode // batch_size)
# steps_per_epoch = 5*val_steps_per_epoch                # number of training batches per epoch (default: 100)

if debug == True: 
    epochs = 5
    steps_per_epoch = 2
    val_steps_per_epoch = 1

# vxm_model.summary()

hist = vxm_model.fit(train_generator, 
                     epochs=epochs, 
                     steps_per_epoch=steps_per_epoch, 
                     verbose=1,
                     validation_data=val_generator,
                     validation_steps=val_steps_per_epoch,
                     callbacks=[reduce_lr, early_stop]) #, tqdm_progress])
    
vxm_model.save_weights(prefix + ".weights.h5")

plot_history(hist)