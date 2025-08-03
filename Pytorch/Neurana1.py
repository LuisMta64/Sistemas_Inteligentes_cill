import torch.nn as nn
import torch
# Crea una neurona: 3 entradas, 1 salida
neuron = nn.Linear(in_features=3, out_features=1)

# Input: tensor de 3 valores
x = torch.tensor([1.0, 2.0, 3.0])
output = neuron(x)

print("Salida de la neurona:", output)