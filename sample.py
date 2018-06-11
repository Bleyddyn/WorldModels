#python 03_generate_rnn_data.py

from vae.arch import VAE
import argparse
import config
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def sample_data(args):
    data_dir = "data"
    env_name = "car_racing"
    data_count = len(glob.glob1(data_dir,"obs_data_{}_*.npy".format(env_name)))

#>>> obs = np.load('data/obs_data_car_racing_0.npy')
#>>> obs.shape
#(200, 300, 64, 64, 3)
    n = args.count
    samples = []

    fname = os.path.join(data_dir,"obs_data_{}_{}.npy".format( env_name, np.random.randint(data_count) ))
    print( "Loading data from {}".format( fname ) )
    obs = np.load(fname)
    for i in range(n):
        ep = np.random.randint(obs.shape[0])
        fr = np.random.randint(obs.shape[1])
        samples.append( obs[ep,fr,:,:,:] )

    input_dim = samples[0].shape
    plt.figure(figsize=(20, 4))
    plt.suptitle( "Generated Data", fontsize=16 )
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(samples[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def sample_vae(args):
    vae = VAE(input_dim=(120,120,3))

    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("./vae/weights.h5 does not exist - ensure you have run 02_train_vae.py first")
      raise

    z = np.random.normal(size=(args.count,vae.z_dim))
    samples = vae.decoder.predict(z)
    input_dim = samples.shape[1:]

    n = args.count
    plt.figure(figsize=(20, 4))
    plt.title('VAE samples')
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(samples[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    #plt.savefig( image_path )
    plt.show()

def main(args):

    if args.data:
        sample_data(args)
    if args.vae:
        sample_vae(args)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Sample one or more stages of training'))
  parser.add_argument('--data', action="store_true", default=False, help='Generate image samples from generated data')
  parser.add_argument('--vae', action="store_true", default=False, help='Generate image samples from a trained VAE')
  parser.add_argument('--count', type=int, default=10, help='How many samples to generate')

  args = parser.parse_args()

  main(args)
