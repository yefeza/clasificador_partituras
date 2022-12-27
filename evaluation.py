from dataset import DatasetLoader
from modeling import build_model
import numpy as np

def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score

def evaluate():
    dataset_loader = DatasetLoader()
    test_data, _ = dataset_loader.read_for_evaluation()
    model = build_model(output_length=_.shape[1], sequence_length=test_data.shape[1])
    # Load model
    model.load_weights("models/model.h5")
    # get predictions
    predictions = model.predict(test_data)
    # get accuracy
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(_, axis=1))
    print("Accuracy: ", accuracy)
    # print predictions converted to integers
    print(np.argmax(predictions, axis=1))
    # compute inception score
    is_score = calculate_inception_score(predictions)
    print("Inception score: ", is_score)

if __name__ == "__main__":
    evaluate()