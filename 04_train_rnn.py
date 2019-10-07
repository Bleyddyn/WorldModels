#python 04_train_rnn.py --new_model

from rnn.arch import RNN
import argparse
import numpy as np

def load_data(fname='./data/rnn_data.npy'):

    data = np.load(fname)
    # print( "Data shape: {}".format( data.shape ) )
    # Data shape: (43212, 131)
    # time, angle, throttle, z
    # record = np.insert( z, 0, [rtime, angle, throttle])
    # Ignoring time for now, will need to use it to not try to learn across time gaps

    rnn_input = data[1:, 1:]
    rnn_output = data[:-1, 3:]
    print( "RNN Input : {}".format( rnn_input.shape ) )
    print( "RNN Output: {}".format( rnn_output.shape ) )
    return rnn_input, rnn_output

def main(args):
    
    new_model = args.new_model

    rnn = RNN( z_dim=128, action_dim=2)

    if not new_model:
        try:
          rnn.set_weights('./rnn/weights.h5')
        except:
          print("Either set --new_model or ensure ./rnn/weights.h5 exists")
          raise

    print( "Loading data..." )
    rnn_input, rnn_output = load_data()

    rnn.train(rnn_input, rnn_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    #parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
    #parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')

    args = parser.parse_args()

    main(args)
