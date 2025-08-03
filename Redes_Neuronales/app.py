import torch

device = 'cpu'
class HebNeuron():
    def __init__(self, n_inputs, device):
        self.device = device
        self.w = torch.zeros((1, n_inputs), dtype=torch.float32, device=device)
        self.b = torch.zeros((1, 1), dtype=torch.float32, device=device)
    def forward(self, x):
        y_in = self.b + torch.matmul(self.w, x.transpose(0, 1))
        y = torch.where(y_in >= 0, 1.0, -1.0)
        return y

    def train(self, inputs, outputs):
        for i in range(len(inputs)):
            x = inputs[i]
            y = outputs[i]
            deltaW = x * y
            deltaB = y

            self.w = self.w + x * y
            self.b = self.b + y
            print(f"Input: { x.to('cpu').numpy() }, Target { y.to('cpu').numpy() } DeltaW: { deltaW.to('cpu').numpy() }, deltaB: { deltaB.to('cpu').numpy() }, currentW: { self.w.to('cpu').numpy() }, currentB: {self.b.to('cpu').numpy()}")

inputs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float32)
outputs = torch.tensor([[1], [-1], [-1], [-1]], dtype=torch.float32) 

hebneuron = HebNeuron(2, device=device)
hebneuron.train(inputs, outputs)





class Perpeptron():
    def __init__(self, n_inputs, threshold, device):
        self.device = device
        self.w = torch.zeros((1,n_inputs), dtype=torch.float32, device=device)
        self.b = torch.zeros((1,1), dtype=torch.float32, device=device)
        self.threshold = threshold
    def forward(self,x):
        y_in = self.b + torch.matmul(self.w,x.transpose(0,1))
        y = torch.where(y_in > self.threshold, 1.0, torch.where(y_in < -self.threshold, -1.0, 0.0))
        return y
    def train(self, inputs, outputs, alpha=0.1, max_epochs=1000, show_info=True):
        epoch=0
        stop_condition = False
        while not stop_condition:
            stop_condition = True
            for i in range(len(inputs)):
                x = inputs[i].unsqueeze(0)
                t = outputs[i]
                y = self.forward(x)
                if y != t:
                    self.w = self.w + alpha * t * x
                    self.b = self.b + alpha + t
                    stop_condition = False
            epoch += 1
            # if show_info 
            if epoch <= max_epochs:
                print("max epochs reached")
                stop_condition: True
        


# print("Weights:", hebneuron.w)
# print("Bias:", hebneuron.b)
# for i in range(len(inputs)):
#     print(f"Input: {inputs[i].tolist()}, Output: {hebneuron.forward(inputs[i].unsqueeze(0))}")