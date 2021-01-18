import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        ############YOUR CODE HERE####################


        ##############################################

    def forward(self, input, hidden):
        ############YOUR CODE HERE####################


        ##############################################
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
