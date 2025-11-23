import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

#TODO: tinker with hyperparameters below to find best and also we should track the outcomes like what enzo put cos it makes it easier to tinker as well
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# get the exact state from the environment's observation (throwback to ccprog1) but only for distance reward
def get_state(state):
     
    bot_row = state // 1000
    bot_col = (state // 100) % 10
    cat_row = (state // 10) % 10
    cat_col = state % 10
    
    return bot_row, bot_col, cat_row, cat_col

#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    # best epsilon = 1.0 up to 0.1
    #Default values for now
    alpha_start = 0.5
    alpha_min = 0.2
    alpha_decay = 0.999
    gamma = 0.9 # discount factor
    epsilon = 1.0 # randomness of exploration

    epsilon_decay = 0.992 # decreasing rate of epsilon
    min_epsilon = 0.05 # minimum exploration rate 
    max_steps = 60 # max steps per episode cos the bot might not reach the cat at all

    outcomes = [] #idk how to use this yet pero nandito na to kanina so di ko na tinanggal
    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        state, _ = env.reset()
        
        total_rewards = 0
        done = False
        
        #epsilon greedy action selection
        
        for _ in range(max_steps):
            
            if random.random() < epsilon:
                action = env.action_space.sample() # explore
                
            else:
                action = np.argmax(q_table[state]) # exploit
                
            next_state, _, done, truncated, _ = env.step(action) # reward is always 0 in the step function so we can ignore it
            
            done = done or truncated # we reached the cat or max steps reached
            
            if done:
                reward = 100 # reached the cat
            else: 
                reward = -1 
                
                #manhattan
                ar, ac, cr, cc = get_state(state)
                ar2, ac2, cr2, cc2 = get_state(next_state)
                dist_before = abs(ar - cr) + abs(ac - cc)
                dist_after = abs(ar2 - cr2) + abs(ac2 - cc2)
                if dist_after < dist_before: 
                    reward += 1.0
                elif dist_after > dist_before: 
                    reward -= 1.0
                    
    
            print("Cat State: ", cr, cc, "Bot State: ", ar, ac, "Reward: ", reward)    
            
            if random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            
            #sarsa algo
            q_table[state][action] = q_table[state][action] + alpha_start * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
            
            total_rewards += reward
            state = next_state
            
            if done:
                break
                
        # decrease epsilon and alpha
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        alpha_start = max(alpha_min, alpha_start * alpha_decay)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table
