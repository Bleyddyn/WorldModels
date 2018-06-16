from __future__ import print_function
import os
import sys
import pickle
from time import time
import argparse

import numpy as np

from keras.utils import to_categorical, Sequence
# https://keras.io/utils/#sequence

#from load_aux import loadAuxData

# For python2/3 compatibility when calling isinstance(x,basestring)
# From: https://stackoverflow.com/questions/11301138/how-to-check-if-variable-is-string-with-python-2-and-3-compatibility
try:
  basestring
except NameError:
  basestring = str

def describeDriveData( data ):
    print( data.keys() )
    for key, value in data.items():
        try:
            print( "{} length {}".format( key, len(value) ) )
        except:
            pass

def embedActions( actions ):
    embedding = { "stop":0, "forward":1, "left":2, "right":3, "backward":4 }
    emb = []
    prev_act = 0
    for act in actions:
        if not act.startswith("speed"):
            prev_act = embedding[act]
            if prev_act is None:
                print( "Invalid action: {}".format( act ) )
                raise ValueError("Invalid action: " + str(act) )
            emb.append( embedding[act] )
        else:
            emb.append( prev_act )
    return emb

def loadOneDrive( drive_dir, size=(120,120), skip_actions=False ):
    if skip_actions:
        actions = None
    else:
        actions_file = os.path.join( drive_dir, "image_actions.npy" )
        if os.path.exists(actions_file):
            actions = np.load(actions_file)
        else:
            actions_file = os.path.join( drive_dir, "image_actions.pickle" )
            with open(actions_file,'r') as f:
                actions = pickle.load(f)

    basename = "images_{}x{}".format( size[0], size[1] )
    im_file = os.path.join( drive_dir, basename+".npy" )
    if os.path.exists(im_file):
        images = np.load(im_file)
    else:
        im_file = os.path.join( drive_dir, basename+".pickle" )
        with open(im_file,'r') as f:
            images = pickle.load(f)
            images = np.array(images)

    images = images.astype(np.float)

    if not skip_actions:
        if images.shape[0] != actions.shape[0]:
            print( "Data mismatch in {}: {} != {}".format( drive_dir, images.shape[0], actions.shape[0] ) )

    return images, actions

def normalize_images( images, default=True ):
    if default:
        rmean = 92.93206363205326
        gmean = 85.80540021330793
        bmean = 54.14884297660608
        rstd = 57.696159704394354
        gstd = 53.739380109203445
        bstd = 47.66536771313241

        print( "Default normalization" )
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd
    else:
        rmean = np.mean(images[:,:,:,0])
        gmean= np.mean(images[:,:,:,1])
        bmean= np.mean(images[:,:,:,2])
        rstd = np.std(images[:,:,:,0])
        gstd = np.std(images[:,:,:,1])
        bstd = np.std(images[:,:,:,2])
        print( "Image means: {}/{}/{}".format( rmean, gmean, bmean ) )
        print( "Image stds: {}/{}/{}".format( rstd, gstd, bstd ) )

# should only do this for the training data, not val/test, but I'm not sure how to do that when Keras makes the train/val split
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd

def postproc_actions( actions ):
    categorical = True
    if not skip_actions:
        if isinstance(actions[0], basestring):
            actions = np.array(actions)
            actions = actions.astype('str')
            actions = embedActions( actions )
            actions = to_categorical( actions, num_classes=5 )
            categorical = True
        elif type(actions) == list:
            actions = np.array(actions)
            categorical = False
        else:
            print("Unknown actions format: {} {} as {}".format( type(actions), actions[0], type(actions[0]) ))

    return actions, categorical

def loadData( dirs, size=(120,120), image_norm=True, skip_actions=False ):
    images = []
    actions = []

    count = 1
    for onedir in dirs:
        if len(onedir) > 0:
            dimages, dactions = loadOneDrive( onedir, size=size, skip_actions=skip_actions )
            images.extend(dimages)
            if not skip_actions:
                actions.extend(dactions)
            print( "Loading {} of {}: {} total samples".format( count, len(dirs), len(images) ), end='\r' )
            sys.stdout.flush()
            count += 1

    print("")
    images = np.array(images)

    if image_norm:
        normalize_images(images)

    categorical = True
    if not skip_actions:
        actions, categorical = postproc_actions(actions)

    return images, actions, categorical

def loadDataBatches( dirs, size=(120,120), image_norm=True, skip_actions=False, max_batch=20000 ):
    images = []
    actions = []

    for idx, onedir in enumerate(dirs):
        if len(onedir) > 0:
            dimages, dactions = loadOneDrive( onedir, size=size, skip_actions=skip_actions )
            images.extend(dimages)
            if not skip_actions:
                actions.extend(dactions)
            print( "Loading {} of {}: {} total samples".format( idx+1, len(dirs), len(images) ), end='\r' )
            sys.stdout.flush()
            if len(images) > max_batch or idx == (len(dirs) - 1):
                print("")
                images = np.array(images)

                if image_norm:
                    normalize_images(images)

                categorical = True
                if not skip_actions:
                    actions, categorical = postproc_actions(actions)

                yield images, actions, categorical
                images = []
                actions = []

class DriveGenerator(Sequence):
    """Generates MaLPi drive data for Keras.
        From: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html"""
    def __init__(self, drive_dirs, batch_size=32, shuffle=True):
        """ Input a list of drive directories.
            Pre-load each to count number of samples.
            load one directory and use it to generate batches until we run out.
            load the next directory, repeat
            Re-shuffle on each epoch end
        """
        'Initialization'
        self.dirs = drive_dirs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.count, self.counts = self.__count()
        self.current_dir = None
        self.current_data = None

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, index):
        sample = index * self.batch_size
        for idx, count in enumerate(self.counts):
            if sample > count:
                continue
            idir = self.dirs[idx]
            if self.current_dir != idir:
                images, y = loadOneDrive(idir, skip_actions=True)
                self.current_data = images
                self.current_dir = idir

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.dirs)

    def __count(self):
# image_actions should have the same count as images, but should be faster to load and count
        count = 0
        counts = []
        for idx, drive_dir in enumerate(self.dirs):
            actions_file = os.path.join( drive_dir, "image_actions.npy" )
            if os.path.exists(actions_file):
                actions = np.load(actions_file)
            else:
                actions_file = os.path.join( drive_dir, "image_actions.pickle" )
                with open(actions_file,'r') as f:
                    actions = pickle.load(f)
            count += len(actions)
            counts.append( count )
        return count, counts

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def runTests(args):
    images, y, cat = loadData(args.dirs)
    print( "Images: {}".format( len(images) ) )
    print( "Actions: {}".format( len(y) ) )
    print( "Actions: {}".format( y[0:5] ) )

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one directory per line')
    parser.add_argument('--aux', default=None, help='Use this auxiliary data in place of standard actions')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            tmp_dirs = f.read().split('\n')
            args.dirs.extend(tmp_dirs)

    if len(args.dirs) == 0 and not args.test_only:
        parser.print_help()
        print( "\nNo directories supplied" )
        exit()

    for i in reversed(range(len(args.dirs))):
        if args.dirs[i].startswith("#"):
            del args.dirs[i]
        elif len(args.dirs[i]) == 0:
            del args.dirs[i]

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    if args.aux is not None:
        auxData, cat_aux = loadAuxData( args.dirs, args.aux )

    skip_actions = True
    images, y, cat = loadData(args.dirs, skip_actions=skip_actions)

    if args.aux is not None:
        y = auxData
        cat = cat_aux
    input_dim = images[0].shape
    if skip_actions:
        num_actions = 0
    else:
        num_actions = len(y[0])
    num_samples = len(images)

    print( "input_dim: {}".format( input_dim ) )
    print( "Action space: {}".format( "Categorical" if cat else "Continuous" ) )
    print( "num_actions: {}".format( num_actions ) )
    print( "num_samples: {}".format( num_samples ) )


    print( "" )
    for images, y, cat in loadDataBatches( args.dirs, skip_actions=skip_actions, max_batch=2000 ):
        input_dim = images[0].shape
        if skip_actions:
            num_actions = 0
        else:
            num_actions = len(y[0])
        num_samples = len(images)
        print( "input_dim: {}".format( input_dim ) )
        print( "Action space: {}".format( "Categorical" if cat else "Continuous" ) )
        print( "num_actions: {}".format( num_actions ) )
        print( "num_samples: {}".format( num_samples ) )
