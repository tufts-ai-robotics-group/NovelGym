import numpy as np
import json
from . import params
from .params import *

from .policy_gradient import RegularPolicyGradient


# from help import *
import pathlib, os

class DiscoverExecutor(object):

    def __init__(
            self, 
            failed_action, 
            action_set, 
            novel_action_set, 
            observation_space, 
            RLModule=RegularPolicyGradient,
            log_every=100
        ) -> None:
        self.log_dir = str(pathlib.Path(__file__).parent.resolve() / 'policies')
        self.log_every = log_every
        self.action_set = action_set
        self.observation_space = observation_space
        self.failed_action = failed_action
        self.play = False
        self.steps = 0
        self.episode = 0
        self.R = []
        self.dones = []
        self.action_count = {}
        self.action_history = []
        self.action_set_dict = dict(zip(self.action_set, range(len(self.action_set))))  # convert the action set to a dictionary.
        self.novel_action_set_dict = {}
        # print ("failed action: {}".format(self.failed_action))
        # TODO: Clean it and technically the env utils should send the dicts of action_set and novel_action_set.
        for key in self.action_set_dict.keys():
            if key in novel_action_set:
                self.novel_action_set_dict.update({key: self.action_set_dict[key]})
        # print(self.novel_action_set_dict)
        print("action set dict:")
        print(json.dumps(self.action_set_dict))
        # print("novel action set dict:")
        # print(json.dumps(self.novel_action_set_dict))
        # get the key,value pair for the failed action to be added to the novel action set dict
        # self.novel_action_set_dict.update({self.failed_action: self.action_set_dict[self.failed_action]})
        
        # initialize the agent
        self.learning_agent = RegularPolicyGradient(
            num_actions=int(len(self.action_set)),
            input_size=int(self.observation_space.shape[0]), 
            hidden_layer_size=NUM_HIDDEN,
            learning_rate=LEARNING_RATE, 
            gamma=GAMMA, 
            decay_rate=DECAY_RATE,
            greedy_e_epsilon=MAX_EPSILON, 
            actions_id=self.action_set_dict,
            random_seed=random_seed, 
            actions_to_be_bumped=self.novel_action_set_dict,
            exploration_mode='ucb',
            guided_action=True, 
            verbose=(log_every == 1)
        )
        self.learning_agent.set_explore_epsilon(MAX_EPSILON)
        # print ("action set dict: {}".format(self.action_set_dict))
        # print ("novel action set dict: {}".format(self.novel_action_set_dict))
        self.loads_model(operator_name=self.failed_action)

    def loads_model(self, operator_name):
        if self.learning_agent.load_model(operator_name=operator_name):
            # if the model exists then load it.
            self.play = True
        elif self.learning_agent.load_model(operator_name=operator_name + "_unconverged"):
            # if we loaded the unconverged model then also load the necessary params.
            self.load_unconverged_params()
            self.play = False
        else:
            if self.log_every <= 100:
                print(f"saved model not found for operator \"{operator_name}\". initializing a new model")
            self.play = False

    def step_episode(self, obs):
        '''
        This function is used to step the episode, for each step the agent takes an action.
        It returns the action predicted by the neural network.
        '''
        if self.play: # if the learned policy already exists then play it instead of learning it
            action = self.play_executor(obs)
            return action
        # set the epsilon based on xthe episode number.
        epsilon = params.MIN_EPSILON + (params.MAX_EPSILON - params.MIN_EPSILON) * math.exp(-params.LAMBDA * self.episode)
        self.learning_agent._explore_eps = epsilon
        if (self.steps + 1) % self.log_every == 0:
            print ("    episode-> {} step--> {} epsilon: {}".format(self.episode, self.steps, round(epsilon, 2)))
        action = self.learning_agent.process_step(obs, exploring=True, timestep=self.steps)
        self.steps += 1
        if self.log_every == 1:
            print ("Executing step {}".format(self.steps))
            print("action: {}".format(self.action_set[action]))
        elif (self.steps + 1) % self.log_every == 0:
            # print("obs: ", obs)
            cum_reward = np.sum(self.learning_agent._drs)
            print ("      step: {};  avg. reward: {}; cumulative reward: {}".format(
                self.steps,
                cum_reward / len(self.learning_agent._drs),
                cum_reward,
            ))

        # count the frequency of an action for debugging
        if action in self.action_count:
            self.action_count[action] += 1
        else:
            self.action_count[action] = 1
        self.action_history.append(action)
        return action

    def end_step(self, reward, should_chop=False):
        '''
        This function is called at the end of each step to give rewards.
        '''
        self.learning_agent.give_reward(reward, should_chop=should_chop)

    def modify_reward(self, reward):
        '''
        This function is used to modify the reward based on the action taken.
        '''
        self.learning_agent.modify_reward(reward)
    
    def end_episode(self, reward, success, should_update_reward=False, should_chop=False):
        """
        Ends the episode, giving appropriate reward.
        """
        if reward is not None:
            self.learning_agent.give_reward(reward, should_chop)
        
        if self.log_every <= 100:
            # log debugging info if verbose.
            print ("total reward: {}".format(np.sum(self.learning_agent._drs)))
            print("last 20 actions: {}".format([self.action_set[action] for action in self.action_history[-50:]]))
            print("action history:")
            for action, count in sorted(self.action_count.items(), key=lambda t: t[1], reverse=True):
                try:
                    action_name = self.action_set[action]
                except Exception:
                    action_name = "unknown"
                print ("   {:>3} {:>3} {} [{}]".format(
                    count, 
                    action, 
                    action_name, 
                    action_name in self.novel_action_set_dict
                ))
        self.R.append(int(np.sum(self.learning_agent._drs)))

        self.learning_agent.finish_episode()
        self.learning_agent.reset_action_counter()
        self.episode += 1

        if success:
            self.dones.append(True) # if the agent reaches the subgoal then append True to the dones list. 
        else:
            self.dones.append(False)
        
        
        self.learning_agent.reset_action_counter()
        if self.episode % params.UPDATE_RATE == 0:
            self.learning_agent.update_parameters()
        
        # save the model.
        if self.episode > 20 and self.check_convergence():
            self.learning_agent.save_model(operator_name=self.failed_action)
            self.learned_executor_exists = True
        else:
            self.save_unconverged_params()
            self.learning_agent.save_model(operator_name=self.failed_action + "_unconverged")
    
    def save_unconverged_params(self):
        '''
        saves the parameters of the agent.
        '''
        file_name = f"{self.log_dir}{os.sep}{self.failed_action}_params.json"
        params = {
            "done": self.dones,
            "R": self.R, 
            'episode': self.episode,
        }
        with open(file_name, 'w') as f:
            json.dump(params, f)


    def load_unconverged_params(self):
        """
        Loads the parameters of the agent.
        """
        file_name = f"{self.log_dir}{os.sep}{self.failed_action}_params.json"
        with open(file_name, 'r') as f:
            params = json.load(f)
        self.dones = params["done"]
        self.R = params["R"]
        self.episode = params["episode"]


    def play_executor(self, obs):
        self.learning_agent.load_model(operator_name=self.failed_action)
        action = self.learning_agent.process_step(obs, exploring=False)
        return action

    def check_convergence(self):
        # this checks if we have to stop learning and save a policy
        if np.mean(self.R[-NO_OF_EPS_TO_CHECK:]) > SCORE_TO_CHECK and np.mean(self.R[-10:]) > SCORE_TO_CHECK: # check the average reward for last 70 episodes
            # for future we can write an evaluation function here which runs a evaluation on the current policy.
            if  np.sum(self.dones[-NO_OF_DONES_TO_CHECK:]) > NO_OF_SUCCESSFUL_DONE and np.mean(self.dones[-10:]) > NO_OF_SUCCESSFUL_DONE/NO_OF_DONES_TO_CHECK: # and check the success percentage of the agent > 80%.
                if abs(np.mean(self.dones[-NO_OF_DONES_TO_CHECK:]) - np.mean(self.dones[-10:])) < 0.05:
                    print ("The agent has learned to reach the subgoal")
                    return True
        else:
            return False

# if __name__ == '__main__':
#     DiscoverExecutor()
