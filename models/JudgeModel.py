import math
import pandas as pd
import torch
from torch import nn




class JudgeModel(nn.Module): # judge whether to sample DONE

    def __init__(self):

        super(JudgeModel,self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.imgeemb1 = nn.Linear(512*7*7,4096)
        self.imgeemb2 = nn.Linear(4096,1027)

        self.innermapping = nn.Linear(1027*2,1024)
        self.finalmapping = nn.Linear(1024,2)

        


        
    def forward(self,imageF,GCNF):
        
        imgemb = self.relu(self.imgeemb1(imageF))
        imgemb = self.relu(self.imgeemb2(imgemb))

        newemb = torch.cat((imgemb, GCNF), dim=1)

        emb = self.relu(self.innermapping(newemb))
        out = self.softmax(self.finalmapping(emb))
        
        return out
