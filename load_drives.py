from __future__ import print_function
import os
import sys
import pickle
from time import time
import argparse

import numpy as np

from load_aux import loadAuxData

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

def loadOneDrive( drive_dir, size=(120,120) ):
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

    return images, actions

def loadData( dirs, size=(120,120), image_norm=True ):
    images = []
    actions = []

    count = 1
    for onedir in dirs:
        if len(onedir) > 0:
            dimages, dactions = loadOneDrive( onedir, size=size )
            if dimages.shape[0] != dactions.shape[0]:
                print( "Data mismatch in {}: {} != {}".format( onedir, dimages.shape[0], dactions.shape[0] ) )
            dimages = dimages.astype(np.float)
            images.extend(dimages)
            actions.extend(dactions)
            print( "Loading {} of {}: {} total samples".format( count, len(dirs), len(images) ), end='\r' )
            sys.stdout.flush()
            count += 1

    print("")
    images = np.array(images)
    #images = images.astype(np.float) # / 255.0

    if image_norm:
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

#        rmean = np.mean(images[:,:,:,0])
#        gmean= np.mean(images[:,:,:,1])
#        bmean= np.mean(images[:,:,:,2])
#        rstd = np.std(images[:,:,:,0])
#        gstd = np.std(images[:,:,:,1])
#        bstd = np.std(images[:,:,:,2])
#        print( "Image means: {}/{}/{}".format( rmean, gmean, bmean ) )
#        print( "Image stds: {}/{}/{}".format( rstd, gstd, bstd ) )
#
## should only do this for the training data, not val/test, but I'm not sure how to do that when Keras makes the train/val split
#        images[:,:,:,0] -= rmean
#        images[:,:,:,1] -= gmean
#        images[:,:,:,2] -= bmean
#        images[:,:,:,0] /= rstd
#        images[:,:,:,1] /= gstd
#        images[:,:,:,2] /= bstd

    categorical = True
    if isinstance(actions[0], basestring):
        actions = actions.astype('str')
        actions = embedActions( actions )
        actions = to_categorical( actions, num_classes=5 )
        categorical = True
    elif type(actions) == list:
        actions = np.array(actions)
        categorical = False
    else:
        print("Unknown actions format: {} {} as {}".format( type(actions), actions[0], type(actions[0]) ))

    return images, actions, categorical

def runTests(args):
    arr1 = hparamsToArray( {} )
    print( "default hparams: {}".format( arr1 ) )
    dict1 = hparamsToDict( arr1 )
    arr2 = hparamsToArray( dict1 )
    if arr1 == arr2:
        print( "round trip worked" )
    else:
        print( "{}".format( arr2 ) )
    dict1["dropouts"] = "up"
    dropouts = [0.2,0.3,0.4,0.5,0.6]
    res = hparamsToArray(dict1)
    if dropouts == res[2]:
        print( "Dropouts with 'up' worked" )
    else:
        print( "Dropouts with 'up' did NOT work" )
    print( args.dirs )

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

    images, y, cat = loadData(args.dirs)

    if args.aux is not None:
        y = auxData
        cat = cat_aux
    input_dim = images[0].shape
    num_actions = len(y[0])
    num_samples = len(images)

