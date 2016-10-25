from __future__ import print_function

import numpy as np
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage
from keras.utils.generic_utils import Progbar
from keras.utils.np_utils import to_categorical
from keras import backend as K
from scipy.stats import entropy as KL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage.transform
from PIL import Image
from PIL import ImageEnhance

DATA_DIR = '/media/ktsakalis/a4d4fb01-5367-4f09-a08c-ae5bd57b4cbd/DSG_data'


def write_subm(preds, test_ids, fname='submission.csv'):
    with open(fname,'w') as f:
        f.write('Id,label' + '\n')
        for i in range(len(test_ids)):
            f.write(test_ids[i]+','+str(preds[i]+1) +'\n')

def rotate90_avg(X, preds, model, all_preds = []):
    S = [[0,1,2,3], [1,0,2,3], [0,1,2,3], [1,0,2,3]]
    for r in range(1,4):
        for i in range(len(X)):
            X[i] = np.rot90(X[i], 1)
        rot_preds = model.predict(X, batch_size=350)
        rot_preds = rot_preds[:, S[r]]
        all_preds.append(np.copy(rot_preds))
        preds += rot_preds
    for i in range(len(X)):
        X[i] = np.rot90(X[i], 1)



def visualise(model, X):
    for l in range(len(model.layers)):

        try:
            fig = plt.figure()
            A = get_activations(model, l, X[0:1])[0]
            for c in range(A.shape[-1]):
                ax = fig.add_subplot(np.sqrt(A.shape[-1]),np.sqrt(A.shape[-1]),c+1)
                plt.imshow(A[:,:,c])
        except Exception as e:
            print(e)
            pass
        plt.show()

def consistency_measure(predictions):
    M = len(predictions)
    N = predictions[0].shape[0]
    divergences = np.zeros((M,M, N))
    for i in range(M):
        for j in range(M):
            divergences[i, j, :] = KL(predictions[i].T, predictions[j].T)

    divergences = divergences.sum(axis=0).sum(axis=0)
    return divergences

def dump_images(I, labels, divergence):
    idxs = np.argsort(labels + (divergence - np.min(divergence)/(np.max(divergence) - np.min(divergence)))  )
    I = I[idxs]
    labels = labels[idxs]
    divergence =  divergence[idxs]
    fig = plt.figure()

    with PdfPages('Images.pdf') as pdf:
        for i in range(I.shape[0]):
            if i % 60 == 0:
                pdf.savefig(fig)
                plt.close('all')
                fig = plt.figure()

            ax = fig.add_subplot(6,10,(i % 60)+1)
            ax.imshow(I[i])
            ax.set_title(str(labels[i]) + '_' + "{0:.2f}".format(divergence[i]), fontsize=8)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            #plt.tight_layout()


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])

    return activations[0]

def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter

def augmentation_transform(zoom=(1.0, 1.0),  rotation=0, shear=0, translation=(0, 0)):
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def transforms(X,y, translation_range = 0.1, rotation_range = 4, zoom_range = (1 / 1.1, 1.1),shear_range = (0,0), rots90_list = None ):
    X_aug = np.copy(X)
    y_aug = np.argmax(y,axis=1)

    w,h = X.shape[2], X.shape[1]
    log_zoom_range = [np.log(z) for z in zoom_range]
    M = [[0, 1, 0, 1], [1, 0, 1, 0], [2, 2, 2, 2], [3, 3, 3, 3]]
    img_shape = X[0,:,:,0].shape
    tform_center, tform_uncenter = build_center_uncenter_transforms(img_shape)

    for i in range(len(X)):
        shear = np.random.uniform(*shear_range)
        shift = np.random.uniform(-w*translation_range, w*translation_range)
        rot90 = np.random.randint(0,4)
        if rots90_list != None:
            rots90_list.append(rot90)

        rot = np.random.uniform(-rotation_range, rotation_range)
        zoomx = np.exp(np.random.uniform(*log_zoom_range))
        zoomy = np.exp(np.random.uniform(*log_zoom_range))
        total_rot = rot90*90 + rot


        for c in range(3):
            tform = tform_uncenter + augmentation_transform((zoomx, zoomy), total_rot, shear, shift) + tform_center
            X_aug[i, :, :, c] =  skimage.transform._warps_cy._warp_fast(X_aug[i, :, :, c],tform.params, output_shape=img_shape, mode='constant', order=1).astype('float32')
        y_aug[i] = M[y_aug[i]][rot90]

        if np.random.randint(0,2) > 0:
            X_aug[i] = np.flipud(X_aug[i])
        if np.random.randint(0,2) > 0:
            X_aug[i] = np.fliplr(X_aug[i])


    return X_aug, to_categorical(y_aug)


def rotation_90_augmentation(X, y):
    print('ROT90')
    #progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking
    X_rot = np.copy(X)
    y_rot = np.where(np.copy(y))[1]
    y_rot = np.copy(y_rot)
    #M[cat][rot] = new_cat
    M = [[0, 1, 0, 1], [1, 0, 1, 0], [2, 2, 2, 2], [3, 3, 3, 3]]
    for i in range(len(X)):
        ri = np.random.randint(0,4)
        X_rot[i] = np.rot90(X[i], ri)
        y_rot[i] = M[y_rot[i]][ri]
        #progbar.add(1)

    return X_rot, to_categorical(y_rot)




def shift_augmentation(X, h_range, w_range):
    print('SHIFT')
    #progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_shift = np.copy(X)

    size = X.shape[1:]
    for i in range(len(X)):
        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])
        if X.ndim > 3:
            for j in range(X.shape[-1]):
                X_shift[i, :,:, j] = ndimage.shift(X[i, :,:, j], (h_shift, w_shift), order=0)
        else:
            X_shift[i,:,:] = ndimage.shift(X[i, :,:], (h_shift, w_shift), order=0)

        #progbar.add(1)

    return X_shift

def rotation_augmentation(X, angle_range):
    print('ROT')
    #progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_rot = np.copy(X)

    for i in range(len(X)):
        angle = np.random.randint(-angle_range, angle_range)
        if X.ndim > 3:
            for j in range(X.shape[1]):
                X_rot[i, j] = ndimage.rotate(X[i, j], angle, reshape=False, order=1)
        else:
            X_rot[i,:,:] = ndimage.rotate(X[i, :,:], angle, reshape=False, order=1)
        #progbar.add(1)

    return X_rot

def flip_augmentation(X):
    print('FLIP')
    #progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking
    X_rot = np.copy(X)

    for i in range(len(X)):
        if np.random.randint(0,2) == 0:
            X_rot[i] = np.flipud(X_rot[i])
        if np.random.randint(0,2) == 0:
            X_rot[i] = np.fliplr(X_rot[i])

    return X_rot


def BC_augmentation(X):
    print('BC')
    X_bc = np.copy(X)
    for i in range(len(X)):
        img = Image.fromarray(X_bc[i].astype(np.uint8))
        img = ImageEnhance.Contrast(img).enhance(np.random.random()*0.6+0.7)
        img = ImageEnhance.Brightness(img).enhance(np.random.random()*0.6+0.7)
        X_bc[i] = np.array(img)
    return X_bc