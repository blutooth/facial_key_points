from __future__ import print_function
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from multiprocessing import Process, Queue
import time
from utils import *
from  model import *


def augment_images_async(q, X, y):
    Xa, ya = augment_images(X, y)
    q.put((Xa, ya))
    return 0

def augment_images(X, y, iter = 0):
    transpose = False
    if X.shape[1] == 3:
        X = X.transpose((0,2,3,1))
        transpose = True

    #Xa, ya = transforms(X, y)
    #return Xa, ya


    Xa, ya = rotation_90_augmentation(X, y)
    Xa = flip_augmentation(Xa)
    Xa = shift_augmentation(Xa, h_range = 0.1, w_range = 0.1)
    Xa = rotation_augmentation(Xa, angle_range = 4)

    Xa = BC_augmentation(Xa)

    if transpose:
        X = X.transpose((0,3,1,2))
        Xa = Xa.transpose((0,3,1,2))
    return Xa, ya


y = to_categorical(np.load(os.path.join(DATA_DIR, 'y_train.npy')) -1)
X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))


model, model_name = get_model()
batch_size = 64

print(X.shape)
for l in range(len(model.layers)):
    print(l, model.layers[l].name, get_activations(model, l, X[0:1]).shape)




nb_iter = 600
epochs_per_iter = 1

Q_SIZE = 3
aug_queue = Queue()
tmp_queue= []

ps = []
for __ in range(Q_SIZE):
    p = Process(target=augment_images_async, args=(aug_queue, X, y))
    p.start()
    ps.append(p)

time.sleep(40)


print('-'*50)
print('Training...')
print('-'*50)
min_val_loss = float("inf")
loss_val = []
loss_train = []
acc_train = []
N = X.shape[0]
avg_acc = 0

#Use all training data and early stop based one mean train accuracy
perc_train = 1.0

for i in range(nb_iter):
    print('-'*50)
    print('Iteration {0}/{1}'.format(i + 1, nb_iter))
    print('-'*50)
    q_size = aug_queue.qsize()

    for __ in range(Q_SIZE - q_size):
        p = Process(target=augment_images_async, args=(aug_queue, X, y))
        p.start()


    while aug_queue.qsize() < 1:
        print('WAITING....')
        time.sleep(1)

    X_train_aug, y_train_aug = aug_queue.get()

    X_train, X_val =  (X_train_aug[0:int(N*perc_train)], X_train_aug[int(N*perc_train):])
    y_train, y_val =  (y_train_aug[0:int(N*perc_train)], y_train_aug[int(N*perc_train):])


    X_train_e = X_train
    y_train_e = y_train

    fitted = model.fit(X_train_e, y_train_e, shuffle=True, nb_epoch=epochs_per_iter,
                                   batch_size=batch_size)

    loss_train.append(fitted.history['loss'][-1])
    acc_train.append(fitted.history['acc'][-1])

    mean_train_acc = np.array(acc_train[-5:]).mean()
    acc_string = "%.2f" % mean_train_acc
    if mean_train_acc > 0.90:
        break
    elif mean_train_acc > 0.84:
        model.save_weights(model_name + acc_string, overwrite=True)