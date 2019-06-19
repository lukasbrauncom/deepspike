import numpy as np

class Task:
    """
    A base class for a training task
    
    ...
    
    Methods
    -------
    training_data(batch_size)
        iterate over training data and corresponding labels
    
    validation_data(batch_size)
        iterate over validation data and corresponding labels
    
    test_data(batch_size)
        iterate over test data and corresponding labels
    
    """
    def __init__(self):  
        """Create placeholders for data"""
        self._training_data = np.array([])
        self._training_labels = np.array([])
        self._validation_data = np.array([])
        self._validation_labels = np.array([])
        self._test_data = np.array([])
        self._test_labels = np.array([])
    
    def training_data(self, batch_size):
        """Iterate over training data and corresponding labels
        
        :param batch_size: (int) size of batches
        :returns: a generator returning batches of training data and corresponding labels
        """
        return self._get_batch(self._training_data, self._training_labels, batch_size)
    
    def validation_data(self, batch_size):
        """Iterate over validation data and corresponding labels
        
        :param batch_size: (int) size of batches
        :returns: a generator returning batches of validation data and corresponding labels
        """
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)
    
    def test_data(self, batch_size):
        """Iterate over test data and corresponding labels
        
        :param batch_size: (int) size of batches
        :returns: a generator returning batches of test data and corresponding labels
        """
        return self._get_batch(self._test_data, self._test_labels, batch_size)
    
    def _get_batch(self, data, labels, batch_size):
        # Shuffle data and return batches of data and labels in a generator
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
            
