#python 03_generate_rnn_data.py

from vae.arch import VAE
import argparse
import config
import numpy as np
import matplotlib.pyplot as plt

def sample_vae(args):
    vae = VAE()

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
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(samples[i].reshape(input_dim[0], input_dim[1], input_dim[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    #plt.savefig( image_path )
    plt.show()

def main(args):

    if args.vae:
        sample_vae(args)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Sample one or more stages of training'))
  parser.add_argument('--vae', action="store_true", default=False, help='Generate image samples from a trained VAE')
  parser.add_argument('--count', type=int, default=10, help='How many samples to generate')

  args = parser.parse_args()

  main(args)
