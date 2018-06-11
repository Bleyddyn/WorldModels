#python 02_train_vae.py --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config

from load_drives import loadData

def main(args):

  start_batch = args.start_batch
  max_batch = args.max_batch
  new_model = args.new_model

  vae = VAE()

  

  if not new_model:
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("Either set --new_model or ensure ./vae/weights.h5 exists")
      raise

  for batch_num in range(start_batch, max_batch + 1):
    print('Building batch {}...'.format(batch_num))
    first_item = True

    for env_name in config.train_envs:
      try:
        new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
        if first_item:
          data = new_data
          first_item = False
        else:
          data = np.concatenate([data, new_data])
        print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
      except:
        pass

    if first_item == False: # i.e. data has been found for this batch number
      data = np.array([item for obs in data for item in obs])
      #print( "Data shape: {}".format( data.shape ) )
# Data shape: (30000, 64, 64, 3)
      vae.train(data)
    else:
      print('no data found for batch number {}'.format(batch_num))

def train_on_drives(args):

    data, y, cat = loadData(args.dirs)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('dirs', nargs='*', metavar="Directory", help='A directory containing recorded robot data')
    parser.add_argument('-f', '--file', help='File with one drive data directory per line')

    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            tmp_dirs = f.read().split('\n')
            args.dirs.extend(tmp_dirs)

    if len(args.dirs) == 0 and not (args.test_only or args.start_batch):
        parser.print_help()
        print( "\nNo directories supplied" )
        exit()

    for i in reversed(range(len(args.dirs))):
        if args.dirs[i].startswith("#"):
            del args.dirs[i]
        elif len(args.dirs[i]) == 0:
            del args.dirs[i]

    if len(args.dirs) > 0:
        train_on_drives(args)
    else:
        main(args)
