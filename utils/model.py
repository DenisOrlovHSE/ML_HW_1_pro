import pickle


def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model