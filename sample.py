#python 03_generate_rnn_data.py

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

model_source = "Unknown"
try:
    from vae.arch import VAE
    import config
    model_source = "AppliedDataSciencePartners"
except:
    pass

try:
    from vae.vae import ConvVAE, reset_graph
    model_source = "hardmaru"
except:
    pass

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

def sample_data2(args):
    data_dir = "record"
    filelist = os.listdir(data_dir)

    n = args.count
    samples = []

    fname = np.random.choice(filelist,1)[0]
    fname = os.path.join(data_dir,fname)
    print( "Loading data from {}".format( fname ) )
    obs = np.load(fname)['obs']
    for i in range(n):
        idx = np.random.randint(obs.shape[0])
        #(1000, 64, 64, 3)
        samples.append( obs[idx,:,:,:] )

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
    """ For vae from: https://github.com/AppliedDataSciencePartners/WorldModels.git
    """
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

def sample_vae2(args):
    """ For vae from https://github.com/hardmaru/WorldModelsExperiments.git
    """
    z_size=32
    batch_size=args.count
    learning_rate=0.0001
    kl_tolerance=0.5
    model_path_name = "tf_vae"

    reset_graph()
    vae = ConvVAE(z_size=z_size,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  kl_tolerance=kl_tolerance,
                  is_training=False,
                  reuse=False,
                  gpu_mode=False) # use GPU on batchsize of 1000 -> much faster

    vae.load_json(os.path.join(model_path_name, 'vae.json'))

    z = np.random.normal(size=(args.count,z_size))
    samples = vae.decode(z)
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
        if model_source == "AppliedDataSciencePartners":
            sample_data(args)
        elif model_source == "hardmaru":
            sample_data2(args)
    if args.vae:
        if model_source == "AppliedDataSciencePartners":
            sample_vae(args)
        elif model_source == "hardmaru":
            sample_vae2(args)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Sample one or more stages of training'))
  parser.add_argument('--data', action="store_true", default=False, help='Generate image samples from generated data')
  parser.add_argument('--vae', action="store_true", default=False, help='Generate image samples from a trained VAE')
  parser.add_argument('--count', type=int, default=10, help='How many samples to generate')

  args = parser.parse_args()

  main(args)
