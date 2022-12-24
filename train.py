from dataset import DatasetLoader
from modeling import build_model

def train():
    dataset_loader = DatasetLoader()
    train_data, train_labels, validation_data, validation_labels = dataset_loader.prepare_dataset()
    model = build_model(output_length=train_labels.shape[1], sequence_length=train_data.shape[1])
    model.fit(train_data, train_labels, epochs=100, validation_data=(validation_data, validation_labels))
    # Save model
    model.save("models/model.h5")
if __name__ == "__main__":
    train()