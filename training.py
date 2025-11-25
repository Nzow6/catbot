import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

#TODO: tinker with hyperparameters below
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
def softmax(q_vals, tau):
    preferences = q_vals / tau
    max_pref = np.max(preferences) 
    exp_values = np.exp(preferences - max_pref)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities
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
    alpha_start = 0.7
    alpha_min = 0.4
    alpha_decay = 0.995
    gamma = 0.9 # discount factor
    # max_steps = 60  max steps per episode cos the bot might not reach the cat at all
    tau = 2.0
    tau_decay = 0.995
    min_tau = 0.01

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

        probabilities = softmax(q_table[state], tau)
        action = np.random.choice(env.action_space.n, p=probabilities)

        #softmax selection
        while not done:
                
            next_state, reward, terminated, truncated, info = env.step(action) # perform action
            
            bot_pos = next_state // 100
            cat_pos = next_state % 100
            
            if bot_pos == cat_pos:
                reward = 100 
                done = True
            else: 
                reward = -1 
                
                #manhattan
                ar, ac, cr, cc = get_state(state)
                ar2, ac2, cr2, cc2 = get_state(next_state)
                dist_before = abs(ar - cr) + abs(ac - cc)
                dist_after = abs(ar2 - cr2) + abs(ac2 - cc2)
                if dist_after < dist_before: 
                    reward += 0.5
                elif dist_after > dist_before: 
                    reward -= 0.5
            done = terminated or truncated       
    
            #print("Cat State: ", cr, cc, "Bot State: ", ar, ac, "Reward: ", reward)    
            if not done:
                next_probs = softmax(q_table[next_state], tau)
                next_action = np.random.choice(env.action_space.n, p=next_probs)
                q_table[state][action] = q_table[state][action] + alpha_start * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
                total_rewards += reward
                state = next_state
                action = next_action
            else:
                q_table[state][action] += alpha_start * (reward - q_table[state][action])
            
                
        # decrease tau and alpha
        alpha_start = max(alpha_min, alpha_start * alpha_decay)
        tau = max(min_tau, tau * tau_decay)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            #print('episode', ep)

    return q_table
