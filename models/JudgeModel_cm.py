import math
import pandas as pd
import torch
from torch import nn




class JudgeModelCM(nn.Module): # judge whether to sample DONE

    def __init__(self):

        super(JudgeModelCM,self).__init__()
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.expand1 = nn.Linear(5,32)
        self.expand2 = nn.Linear(32,64)
        
        self.suqeeze1 = nn.Linear(300,128)
        self.suqeeze2 = nn.Linear(128,64)

        self.suqeeze3 = nn.Linear(512*7*7,1024)
        self.suqeeze4 = nn.Linear(1024,1024)
        self.suqeeze5 = nn.Linear(1024,512)
        self.suqeeze6 = nn.Linear(512,64)

        self.expand3 = nn.Linear(1,10)

        self.finalmapping = nn.Linear(64*3 + 10,2)
        #self.finalmapping2 = nn.Linear(256,2)

        


        
    def forward(self,feature):

        

        CM = feature[:,:5] # split for cm

        Glove = feature[:,5:305] # split for word embedding

        frame = feature[:,305:512*7*7 + 305]

        num_objects = feature[:,-1:]


        
        
        CM_out = self.relu( self.expand1(CM))
        CM_out = self.relu( self.expand2(CM_out))

        Glove_out = self.relu( self.suqeeze1(Glove))
        Glove_out = self.relu( self.suqeeze2(Glove_out))

        frame_out = self.relu( self.suqeeze3(frame))
        frame_out = self.relu( self.suqeeze4(frame_out))
        frame_out = self.relu( self.suqeeze5(frame_out))
        frame_out = self.relu( self.suqeeze6(frame_out))

        num_objects_out = self.relu( self.expand3(num_objects))



        cat = torch.cat((CM_out, Glove_out, frame_out, num_objects_out), dim=1)

        out = self.finalmapping(cat)

        out = self.softmax(out) 
        
        return out