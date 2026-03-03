from HelperFuncs import PredictiveChanger
import torch

if __name__ == '__main__':
    model = PredictiveChanger(400,400,21,1e6)
    torch.save(model,'models/initial.pt')
