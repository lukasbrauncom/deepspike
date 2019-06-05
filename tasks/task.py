import numpy as np

class Task:
    def __init__(self):
        self._training_data = []
        self._training_labels = []
        self._validation_data = []
        self._validation_labels = []
        self._test_data = []
        self._test_labels = []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.step >= self.max_steps:
            raise StopIteration
        self.step += 1
        return self.inputs[self.step-1], self.outputs[self.step-1]
    
    def training_data(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)
    
    def validation_data(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)
    
    def test_data(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)
    
    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]
        if batch_size <= 0:
            batch_size = samples_n
        
        random_indices = np.random.choice(samples_n, samples_n, replace = False)
        data = data[random_indices]
        labels = labels[random_indices]
        
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]
            
