import numpy as np
from .task import Task

class IntegratorTask(Task):
    """
    A temporal task for integrator neurons, inherits from Task
    
    ...
    
    Attributes
    ----------
    input_neurons_n : int
        amount of input spike trains
    length : int
        length of spike trains
    snr : float
       proportion of signal carrying to noisy spike trains
    
    Methods
    -------
    training_data(batch_size)
        iterate over training data and corresponding labels
    
    validation_data(batch_size)
        iterate over validation data and corresponding labels
    
    test_data(batch_size)
        iterate over test data and corresponding labels
    """
    def __init__(self, input_neurons_n, length, snr=1, training_size=10000, validation_size=1000, test_size=5000):
        """Create semi-random training, validation and test data
        
        :param input_neurons: (int) amount of input spike trains
        :param length: (int) length of spike trains
        :param snr: (float) proportion of signal carrying to noisy spike trains
        :param training_size: (int) amount of training samples
        :param validation_size: (int) amount of validation samples
        :param test_size: (int) amount of test samples
        """
        self.input_neurons_n = input_neurons_n
        self.length = length
        self.snr = snr
        
        # Create data buffers
        self._training_data = np.empty((training_size, input_neurons_n, length))
        self._training_labels = np.empty((training_size, length))
        self._validation_data = np.empty((validation_size, input_neurons_n, length))
        self._validation_labels = np.empty((validation_size, length))
        self._test_data = np.empty((test_size, input_neurons_n, length))
        self._test_labels = np.empty((test_size, length))
        
        # Generate data
        # Ensure semi-randomness for comparability by setting seeds
        np.random.seed(1)
        self._generate_data(self._training_data, self._training_labels, training_size)
        
        np.random.seed(2)
        self._generate_data(self._validation_data, self._validation_labels, validation_size)
        
        np.random.seed(3)
        self._generate_data(self._test_data, self._test_labels, test_size)

    def _generate_data(self, data, labels, samples_n):
        # Sample i spike trains, where n carry information and r are noisy
        for i in range(samples_n):
            firing_prob, labels[i, :] = self._firing_prob()
            for n in range(int(np.round(self.input_neurons_n*self.snr))):
                data[i, n, :] = self._generate_spikes(firing_prob)
            for r in range(int(np.round(self.input_neurons_n*(1-self.snr)))):
                sample[i, n+1+r, :] = self._generate_spikes(np.ones(length)*0.05)
    
    def _generate_spikes(self, firing_prob):
        # Generate spike train, using a firing probability
        spike_train = np.zeros(self.length)
        spike_train[firing_prob > np.random.rand(self.length)] = 1
        return spike_train

    def _firing_prob(self):
        # Generate a firing probability by convolving a spike train 
        x = np.linspace(-15, 15, 300)
        gauss = self._gauss(x, 0, 4)
        events = np.random.choice([0, 1], self.length, p=[0.99, 0.01])    
        firing_prob = np.convolve(gauss, events, "same")*0.3
        labels = np.roll(events, int(4 * (300/(15+15))))
        labels[:int(4 * (300/(15+15)))] = 0
        return firing_prob, labels
    
    def _gauss(self, x, mu, sigma):
        # Return a Gaussian probability density
        return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
    
    
    
    
