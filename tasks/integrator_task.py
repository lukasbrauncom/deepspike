import numpy as np

from .task import Task

class IntegratorTask(Task):
    def __init__(self, input_neurons_n, length, snr=1):
        self.input_neurons_n = input_neurons_n
        self.length = length
        self.snr = snr
        
        sample = np.zeros((input_neurons_n, length))
        
        firing_prob, labels = self._firing_prob()
        for i in range(int(np.round(input_neurons_n*snr))):
            sample[i, :] = self._generate_spikes(firing_prob)
        for n in range(int(np.round(input_neurons_n*(1-snr)))):
            sample[i+1+n, :] = self._generate_spikes(np.ones(length)*0.05)
        
        self._training_data = np.asarray([sample])
        self._training_labels = np.asarray([firing_prob])
            
    
    def _generate_spikes(self, firing_prob):
        spike_train = np.zeros(self.length)
        spike_train[firing_prob > np.random.rand(self.length)] = 1
        return spike_train

    def _firing_prob(self):
        x = np.linspace(-15, 15, 300)
        gauss = self._gauss(x, 0, 4)
        events = np.random.choice([0, 1], self.length, p=[0.99, 0.01])    
        firing_prob = np.convolve(gauss, events, "same")*0.3
        labels = np.roll(events, int(4 * (300/(15+15))))
        labels[:int(4 * (300/(15+15)))] = 0
        return firing_prob, labels
    
    def _gauss(self, x, mu, sigma):
        return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
