# SPECIFICATIONS:
# Write interface which:
# Trains classifier on original dataset
# Performs OOSE
# Runs classifier on OOSE points
import numpy as np
import numpy.typing as npt
import sklearn as sk
import torch
from torch.utils.data import Dataset
from torchvision import datasets as datasetsta
from torch.utils.data import DataLoader

class Classifier:
    """
    Classifier interface:\n
    All classifiers passed to ClassificationTests should inherit this class\n
    Classes implementing this interface should specify learning rate as a instance varialbe (lr: float)
    """
    def classify(self, inputs: torch.Tensor):
        """
        Params: inputs\n
        Run the classifier model on some inputs
        """
        pass
        
    def train(self, data: npt.NDArray[Any], training_ind: npt.NDArray[np.int_]) -> None:
        """
        Params: data, training indices\n
        Return: None
        """
        pass

class GenericDataLoader:
    def __init__(self, data: npt.NDArray[Any], labels: npt.NDArray[Any], training_ind: npt.NDArray[np.int_],
                 batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> None:
        self.data = data[training_ind]
        self.labels = labels[training_ind]
        self.training_ind = training_ind
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.__dataset()

    def get_data(self) -> npt.NDArray[Any]:
        return self.data
    
    def __dataset(self) -> None:
        self.dataset = self.GenericDataset(self.data, self.labels)

    def dataset(self):
        return self.dataset
    
    def dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=self.shuffle, num_workers=self.num_workers)
        
    class GenericDataset(Dataset):
        def __init__(self, data: npt.NDArray[Any], labels: npt.NDArray[Any], transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform

        def __getitem__(self, index):
            return self.data[index], self.labels[index]
        
        def __len__(self) -> int:
            return len(self.labels)
        
class NN(torch.nn.Module):
    def __init__(self):
        super.__init__()
        self.layer1 = torch.nn.Linear(120, 64)
        self.layer2 = torch.nn.Linear(64, 32)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return x
    
class NNClassifier(Classifier):
    def __init__(self):
        self.net = NN()
        self.lr = 0.1

#     def classify(self, inputs: torch.Tensor):
#         self.net(inputs)
    
#     def train(self, data: Dataset, training_ind: npt.NDArray[np.int_]) -> None:
#         mse = torch.nn.MSELoss()
#         # define inputs and labels from datasets\

#         # define train loop

#         optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
#         optimizer.zero_grad()

#         outputs = self.net(inputs)

#         loss = mse(outputs, labels)

#         loss.backward()

#         optimizer.step()

class ClassificationTests:
    def __init__(self, data: npt.NDArray[Any], training_ind: npt.NDArray[np.int_], classifier: Classifier) -> None:
        self.classifier = classifier
        self.data = data
        self.training_ind = training_ind
        self.classifier.train(self.data, self.training_ind)

    # def __get_samples(self, ind: npt.NDArray[np.int_]) -> npt.NDArray[Any]:
    #     return np.ndarray([self.data.__getitem_(i) for i in ind])

    def compare(self) -> float:
        # perform OOSE, store classification in list
        # samples = self.__get_samples(self.training_ind)
        pred = None # Will be classify(OOSE)
        pred_true = self.classifier.classify(self.data)



    
