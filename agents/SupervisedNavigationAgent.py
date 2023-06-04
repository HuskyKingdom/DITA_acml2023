import torch
from utils.net_util import gpuify
from models.model_io import ModelInput
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.constants import DONE_ACTION_INT

from .agent import ThorAgent
import queue
import time
import random

from tensorboardX import SummaryWriter

class SupervisedNavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, NB_model,NB_optimizer,NB_criterion,gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(SupervisedNavigationAgent, self).__init__(
            create_model(args), args, rank, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz


        

        self.supervision = []
        self.batchbuffer = []
        self.batchMaxSize = 64
        self.nb_model = NB_model
        self.nb_optm = NB_optimizer
        self.nb_cri = NB_criterion
        self.rank = rank

        self.numDone = 0
        self.numNDone = 0

        self.preR = -0.01

        # NB 
        start_time = time.time()
        local_start_time_str = time.strftime(
        "%Y-%m-%d_%H:%M:%S", time.localtime(start_time)
        )

        self.args = args

        if self.rank == 0 and not self.args.eval:
            self.writer = SummaryWriter('logs/NewNB_full_'+local_start_time_str)
        
        self.i = 0

        

        if self.args.eval: # load the model if testing

            try:
                saved_state = torch.load(
                    args.load_JG_model, map_location=lambda storage, loc: storage
                )
                self.nb_model.load_state_dict(saved_state)
            except:
                self.nb_model.load_state_dict(torch.load(args.load_JG_model))

        


    def action(self, model_options, training, demo=False): # rewrite with NB
        """ Train the agent. """
        if training:
            self.model.train()
            self.nb_model.train()
        else:
            self.model.eval()
            self.nb_model.eval()

        # RL inference
        model_input, out = self.eval_at_state(model_options)
 
     


        

        action_index = 1

        # NB inference
        nb_input = out.embedding.clone().detach().cuda()
        nb_out = None

        if not self.args.eval:

            if nb_input[:,:5][0][0] >= 0.9:
                nb_out = self.nb_model(nb_input) # (1,305)
                pro_done, pro_Nodone = nb_out[0][0],nb_out[0][1]
                print("MODEL INFERENCE {} , {}".format(pro_done,pro_Nodone))
                
                # judge model takes part

                
                
                action_index = random.choices([0, 1], nb_out[0], k=1)[0]
            
        else:

            if nb_input[:,:5][0][0] >= 0.9:
                nb_out = self.nb_model(nb_input) # (1,305)
                pro_done, pro_Nodone = nb_out[0][0],nb_out[0][1]


                if nb_out[0][0] > nb_out[0][1]:
                    action_index = 0

                # action_index = random.choices([0, 1], nb_out[0], k=1)[0]
            
        
        
                

        
            



        self.hidden = out.hidden

        prob = F.softmax(out.logit, dim=1)
        action = prob.multinomial(1).data
        log_prob = F.log_softmax(out.logit, dim=1)
        self.last_action_probs = prob
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))
        

       
        pre_state = self.environment.controller.state

        # action control
        
        print(prob)


        if not self.args.eval:


            self.reward, self.done, self.info = self.episode.step(action[0, 0]) 



        else:
        
            if (nb_out != None and nb_out[0][0] + prob[0][5] >= 1.5):

                self.reward, self.done, self.info = self.episode.step(5)
                action = 5


            else:


                if action_index == 0:

                    self.reward, self.done, self.info = self.episode.step(action[0, 0])

                else:
                    
                    max_time = 0
                    while action[0, 0] == 5 and max_time <= 50:
                        action = prob.multinomial(1).data
                        max_time += 1

                    self.reward, self.done, self.info = self.episode.step(action[0, 0]) 
            

        ground_truth = 1
        current_action = action[0, 0]

        if self.reward >= 4:
            ground_truth = 0

        

            
        
        if nb_out != None:
            print("{} | Ground {}".format(nb_out.tolist()[0],ground_truth))
        else:
            print("NONE | Ground {}".format(ground_truth))
        
        if self.args.vis and action == 5:
                print("Success:", self.info)
        
        print("\n")

        self.preR = self.reward
        

        # NB Store and Update if rank == 0 and in training
        if self.rank == 0 and not self.args.eval:
            
            if len(self.batchbuffer) == self.batchMaxSize: # bufferring
                

                # Updating Shared NB Model

                # making inputs and targets

                nb_input = self.batchbuffer[0] # first element

                for i in range(1,len(self.batchbuffer)): # making a batch
                    nb_input = torch.cat((nb_input, self.batchbuffer[i]), dim=0)

                
                nb_target = torch.tensor(self.supervision).cuda()


                # learning
                self.nb_optm.zero_grad()

                nb_output = self.nb_model(nb_input)

                

                
                nb_loss = self.nb_cri(nb_output, nb_target)

                
                
        


                print("*****nb_out_train {}".format(nb_output))
                print("*****nb_true {}".format(nb_target))

                print("*****nb_loss {}".format(nb_loss))

    
                
                self.writer.add_scalar('loss', nb_loss, self.i) # logs
                
                self.i += 1


                
                nb_loss.backward()
                for name , parms in self.nb_model.named_parameters():
                    print("NAME {} | REQUIRED {} | GRADIENT {} | \n\n\n".format(name,parms.requires_grad, parms.grad))

                
                self.nb_optm.step()

            

                # clear
                self.batchbuffer = []
                self.supervision = []
                self.numDone = 0
                self.numNDone = 0

            temp = out.embedding # image + gcn

            


            print("temp {} | ground {} | agent current state {}".format(temp[:,:5],ground_truth,pre_state))

            
            if temp[:,:5][0][0] >= 0.9 and current_action == DONE_ACTION_INT:
                self.batchbuffer.append(temp)
                self.supervision.append(ground_truth)
                print("*****************STORED {} with {} \n\n\n".format(len(self.batchbuffer),ground_truth ))

        
        
        else:
            self.i += 1


            

            

        
        

        if self.verbose:
            pass
            print(self.episode.actions_list[action])
        self.probs.append(prob)
        self.entropies.append(entropy)
        self.values.append(out.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.episode.prev_frame = model_input.state
        self.episode.current_frame = self.state()

        if self.learned_loss:
            res = torch.cat((self.hidden[0], self.last_action_probs), dim=1)
            if self.learned_input is None:
                self.learned_input = res
            else:
                self.learned_input = torch.cat((self.learned_input, res), dim=0)

        self._increment_episode_length()

    

        if self.episode.strict_done and action == DONE_ACTION_INT:
            self.success = self.info
            self.done = True
        elif self.done:
            self.success = not self.max_length


        


        return out.value, prob, action


    def eval_at_state(self, model_options):
        model_input = ModelInput()
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        if self.episode.current_objs is None:
            model_input.objbb = self.objstate()
        else:
            model_input.objbb = self.episode.current_objs
        model_input.hidden = self.hidden
        model_input.target_class_embedding = self.episode.glove_embedding
        model_input.action_probs = self.last_action_probs

        model_input.objects = self.environment.controller.last_event.metadata

        

        return model_input, self.model.forward(model_input, model_options)

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                )
        else:
            self.hidden = (
                torch.zeros(1, self.hidden_state_sz),
                torch.zeros(1, self.hidden_state_sz),
            )
        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )


    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def objstate(self):
        return self.episode.objstate_for_agent()
