from keras.models import load_model
import pickle

iwpod_net = load_model("weights/iwpod.h5")
weights = iwpod_net.get_weights()

with open('weights.pkl', 'wb') as outfile:
    pickle.dump(weights, outfile, pickle.HIGHEST_PROTOCOL)