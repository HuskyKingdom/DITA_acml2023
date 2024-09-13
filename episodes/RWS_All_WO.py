""" Contains the Episodes for Navigation. """
import random

import torch

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY
from datasets.constants import DONE
from datasets.environment import Environment

from utils.net_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.net_util import gpuify
from .episode import Episode
from utils import flag_parser

import json

c2p_prob = json.load(open("./data/c2p_prob.json"))
args = flag_parser.parse_arguments()

class RWS_All_WO(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(RWS_All_WO, self).__init__()

        self._env = None
        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None

        self.scene_states = []
        self.partial_reward = args.partial_reward
        self.seen_list = []
        if args.eval:
            random.seed(args.seed)
        self.room = None
        self.pre_bbox = 0.0
        self.pre_state = None
        self.i = 0 #"count for vis"

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def objstate_for_agent(self):
        return self.environment.current_objs

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int):

        action = self.actions_list[action_as_int]

        # for Judge Model
        ground_truth = 1 # not done
        # print("task data {} ___________________ ".format(self.task_data))
        for id_ in self.task_data: # Target object ['SprayBottle|+00.38|+00.81|+02.12']
            if self.environment.object_is_visible(id_):
                ground_truth = 0 # done
                break


        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        

        reward, terminal, action_was_successful = self.judge(action)

        if args.vis:
            print("{}|{}".format(action,reward))

        # if args.vis and action["action"] == DONE:
        #         print("Success:", action_was_successful)


        

        return reward, terminal, action_was_successful, ground_truth

    def cal_state_dif(self,current_str,other_str):
        current = str(current_str).split("|")
        other = str(other_str).split("|")
        dif = 0.0
        for i in range(len(current)):
            dif += abs(float(current[i]) - float(other[i]))
        return dif

        
    def judge(self, action):
        """ Judge the last event. """
        reward = STEP_PENALTY
        
        # Thresholding replaced with simple look up for efficiency.
        
        if self.environment.controller.state in self.scene_states: # -0.50|1.50|0|30
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
                # added partial reward
                if self.partial_reward:
                    reward = self.get_partial_reward(action)
        else:
            self.scene_states.append(self.environment.controller.state)
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
                # added partial reward
                if self.partial_reward:
                    reward = self.get_partial_reward(action)

        done = False
        
        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data: # Target object ['SprayBottle|+00.38|+00.81|+02.12']
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    
                    # Removing for WO

                    # Stuck Detection

                    if self.pre_state == None:
                        self.pre_state = self.environment.controller.state
                    else:
                        dif = self.cal_state_dif(self.pre_state,self.environment.controller.state)
                        if dif == 0.0:
                            reward -= 0.03
                        self.pre_state = self.environment.controller.state

                    # Increasing bbox area increasing reward

                    # Calculating bbox area
                    t_obj = self.target_object
                    k = list(self.objstate_for_agent().keys())
                    
                    if t_obj in k: # if target is seen
                        area = self.cal_area(self.objstate_for_agent()[t_obj])
                        

                        if area > self.pre_bbox: # if increased bbox
                            test = self.pre_bbox
                            dif = area - self.pre_bbox
                            self.pre_bbox = area
                            reward += 0.00001 * dif

                    break

            self.seen_list = []

        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def get_partial_reward(self,action):
        """ get partial reward if parent object is seen for the first time"""
        reward = STEP_PENALTY
        reward_dict = {}

        # print("current object __________ {}".format(self.objstate_for_agent()))
        # print("target_parents __________ {}".format(self.target_parents))

        if self.target_parents is not None: # {'Drawer': 0.12, 'TableTop': 0.15, 'Shelf': 0.06, 'Sofa': 0.1, 'Television': 0.08}
            
            for parent_type in self.target_parents:

                parent_ids = self.environment.find_id(parent_type) # ['TableTop|-00.29|+00.04|-00.77', 'TableTop|+01.59|00.00|+00.45']
                
                for parent_id in parent_ids:
                    if self.environment.object_is_visible(parent_id) and parent_id not in self.seen_list:
                        reward_dict[parent_id] = self.target_parents[parent_type]

        
        if len(reward_dict) != 0:
            v = list(reward_dict.values())
            k = list(reward_dict.keys())
            reward = max(v)           #pick one with greatest reward if multiple in scene
            self.seen_list.append(k[v.index(reward)])


        # Stuck Detection

        if self.pre_state == None:
            self.pre_state = self.environment.controller.state
        else:
            dif = self.cal_state_dif(self.pre_state,self.environment.controller.state)
            if dif == 0.0:
                reward -= 0.03
            self.pre_state = self.environment.controller.state

        # Increasing bbox area increasing reward

        # Calculating bbox area
        t_obj = self.target_object
        k = list(self.objstate_for_agent().keys())
        
        if t_obj in k: # if target is seen
            area = self.cal_area(self.objstate_for_agent()[t_obj])
            

            if area > self.pre_bbox: # if increased bbox

                dif = area - self.pre_bbox
                self.pre_bbox = area
                reward += 0.0001 * dif
                


        return reward


    def cal_area(self,obj):
        return (obj[2] - obj[0]) * (obj[3] - obj[1])

    def _new_episode(
        self, args, scenes, possible_targets, targets=None, room = None, keep_obj=False, glove=None
    ):
        """ New navigation episode. """
        scene = random.choice(scenes)
        self.room = room
        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        # Randomize the start location.
        start_state = self._env.randomize_agent_location()
        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        child_object = self.task_data[0].split("|")[0]
        #print('room is ', self.room)
        try:
            self.target_parents = c2p_prob[self.room][child_object]
        except:
            self.target_parents = None

        if args.verbose:
            print(self.i,"Scene", scene, "Navigating towards:", goal_object_type)
            self.i += 1

        self.glove_embedding = None
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[goal_object_type][:], self.gpu_id
        )

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        rooms=None,
        keep_obj=False,
        glove=None,
    ):
        self.pre_bbox = 0.0
        self.pre_state = None
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None
        self._new_episode(args, scenes, possible_targets, targets, rooms, keep_obj, glove)
