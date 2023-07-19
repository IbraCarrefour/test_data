import torch
import torch.nn as nn
import torch.optim as optim
from keras import backend as K

K.clear_session()

class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 32 * 16)
        x = self.sigmoid(self.fc(x))

        return x


class BinaryClassifierMLP(nn.Module):
    def __init__(self):

        super(BinaryClassifierMLP, self).__init__()
        self.layer1 = nn.Linear(16, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        x = self.relu(self.layer1(inputs))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))

        return x


def calculate_accuracy(outputs, labels):

    predicted_labels = (outputs >= 0.5).int()
    correct_predictions = (predicted_labels == labels).sum().item()
    accuracy = correct_predictions / labels.size(0)

    return accuracy


def calculate_recall(outputs, labels):

    predicted_labels = (outputs >= 0.5).int()
    true_positives = ((predicted_labels == 1) & (labels == 1)).sum().item()
    false_negatives = ((predicted_labels == 0) & (labels == 1)).sum().item()
    recall = true_positives / (true_positives + false_negatives)

    return recall