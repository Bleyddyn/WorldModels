#python 02_train_vae.py --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config
import os

DIR_NAME = './data/rollout/'
M = 300
SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64


def import_data(N):
  filelist = os.listdir(DIR_NAME)
  filelist = [x for x in filelist if x != '.DS_Store']
  filelist.sort()
  length_filelist = len(filelist)


  if length_filelist > N:
    filelist = filelist[:N]

  if length_filelist < N:
    N = length_filelist

  data = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)
  idx = 0
  file_count = 0


  for file in filelist:
      try:
        new_data = np.load(DIR_NAME + file)['obs']
        data[idx:(idx + M), :, :, :] = new_data

        idx = idx + M
        file_count += 1

        if file_count%50==0:
          print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
      except:
        print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

  return data, N



from load_drives import loadData, loadDataBatches, DriveGenerator

def main(args):

  new_model = args.new_model
  N = int(args.N)
  epochs = int(args.epochs)

  vae = VAE()

  if not new_model:
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("Either set --new_model or ensure ./vae/weights.h5 exists")
      raise

  try:
    data, N = import_data(N)
  except:
    print('NO DATA FOUND')
    raise
      
  print('DATA SHAPE = {}'.format(data.shape))

  for epoch in range(epochs):
    print('EPOCH ' + str(epoch))
    vae.train(data)
    vae.save_weights('./vae/weights.h5')

def train_on_drives(args):
    vae = None

    for data, y, cat in loadDataBatches( args.dirs, skip_actions=True, max_batch=10000 ):
        if vae is None:
            input_dim = data[0].shape
            print( "Data shape: {}".format( data.shape ) )
            vae = VAE( input_dim=input_dim )

            if not args.new_model:
                try:
                    vae.set_weights('./vae/weights.h5')
                except:
                    print("Either set --new_model or ensure ./vae/weights.h5 exists")
                    raise

        vae.train(data, epochs=100)

def train_on_drives_gen(args):
    input_dim=(120,120,3)
    gen = DriveGenerator(args.dirs, input_dim=input_dim, batch_size=32, shuffle=True, max_load=10000, skip_actions=True, image_norm=True)
    val = DriveGenerator(args.val, input_dim=input_dim, batch_size=32, shuffle=True, max_load=10000, skip_actions=True, image_norm=True)
    vae = VAE( input_dim=input_dim )
    print( "Train: {}".format( gen.count ) )
    print( "Test : {}".format( val.count ) )

    if not args.new_model:
        try:
            vae.set_weights('./vae/weights.h5')
        except:
            print("Either set --new_model or ensure ./vae/weights.h5 exists")
            raise

    vae.train_gen(gen, val, epochs=100)

if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description=('Train VAE'))
#    parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
#    parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
#    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
#    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
#    parser.add_argument('-f', '--file', help='File with one drive data directory per line')
#    parser.add_argument('--val', help='File with one drive data directory per line for validation')
#    parser.add_argument('--val_split', type=float, default=0.2, help='Percent validation split')
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
    args = parser.parse_args()

    main(args)

#    if args.file is not None:
#        with open(args.file, "r") as f:
#            tmp_dirs = f.read().split('\n')
#            args.dirs.extend(tmp_dirs)
#
#    if len(args.dirs) == 0 and not (args.test_only or args.start_batch):
#        parser.print_help()
#        print( "\nNo directories supplied" )
#        exit()
#
#    for i in reversed(range(len(args.dirs))):
#        if args.dirs[i].startswith("#"):
#            del args.dirs[i]
#        elif len(args.dirs[i]) == 0:
#            del args.dirs[i]
#
#    if args.val is not None:
#        with open(args.val, "r") as f:
#            tmp_dirs = f.read().split('\n')
#            args.val = tmp_dirs
#        for i in reversed(range(len(args.val))):
#            if args.val[i].startswith("#"):
#                del args.val[i]
#            elif len(args.val[i]) == 0:
#                del args.val[i]
#    else:
#        last = int(len(args.dirs) * args.val_split)
#        np.random.shuffle(args.dirs)
#        test = args.dirs[:last]
#        train = args.dirs[last:]
#        args.dirs = train
#        args.val = test
#        #print( "Train:\n{}".format( args.dirs ) )
#        #print( "Test:\n{}".format( args.val ) )
#
#    if len(args.dirs) > 0:
#        #train_on_drives(args)
#        train_on_drives_gen(args)
#    else:
#        main(args)
