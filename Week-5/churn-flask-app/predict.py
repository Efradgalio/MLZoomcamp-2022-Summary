import pickle


# Load the Model
model_file = f'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
# we don't need to import scikit-learn, but we need scikit-learn installed in our system,
# so it will know what model and dv means.

