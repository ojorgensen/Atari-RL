from general_utils import *
from nets_utils import *
from accessory import print_episode_info, recorder, print_training_info

import time
import torch
import torch.optim as optim
from gyms.cartpole import CartPoleEnv



def train_DQN(type:str='DQN', env_name:str='CartPole', n_runs:int=10, starting_eps:float=1., network_layers:list[int]=[4,2], 
        episode_print_thresh:int=150, n_episodes:int=300, buffer_size=1000, batch_size=1, update_when=1, learning_rate=1, decay=0.99,
        recordings_dir_name:str='episode_recorder', episode_base_name:str='episode', record=False, max_episode_steps=500):
    
    print_training_info(type, env_name, n_runs, starting_eps, n_episodes, decay, network_layers, batch_size, buffer_size, update_when)
    runs_results = []
    if env_name == 'CartPole':
        env = CartPoleEnv(render_mode='rgb_array')
    
    # add other environment options here

    env._max_episode_steps = max_episode_steps
    
    # loop through a run
    for run in range(n_runs):
        if record: video_dir = recorder('new_run', recordings_dir_name=recordings_dir_name, run=run)   # <><><><1><><><> #

        # initialise networks and update scheme
        t0 = time.time()
        policy_net, target_net = initialise_networks(network_layers)
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)          # using Adam gradient descent
        memory = ReplayBuffer(buffer_size)                                         # a replay buffer of size buffer_size

        # loop through episodes
        episode_durations = []
        for i_episode in range(n_episodes):
            print_episode_info(i_episode, n_episodes, episode_print_thresh)
            if record: video_recorder = recorder('start_episode', episode_base_name=episode_base_name, video_dir=video_dir, env=env, i_episode=i_episode)   # <><><><2><><><> #

            # initialise episode starting state
            state, done, terminated, t = initialise_episode(env)

            # generate steps and update through an episode
            while not (done or terminated):
                if record: video_recorder.capture_frame()                                  # <><><><3><><><> #

                # select action, observe results; push to memory
                action, next_state, reward, done, terminated = step_episode(env, policy_net, state, starting_eps, decay, i_episode)
                memory.push([state, action, next_state, reward, torch.tensor([done])])
                state = next_state

                # update the policy net
                update_policy(memory, policy_net, target_net, optimizer, type, batch_size)

                # check state termination
                if done or terminated: episode_durations.append(t+1)
                t += 1

            if record: recorder('end_episode', video_recorder)                     # <><><><4><><><> #

            # update the target net
            update_target(target_net, policy_net, i_episode, update_when)

        runs_results.append(episode_durations)
        t1 = time.time()
        print(f"Ending run {run+1} of {n_runs} with run time: {round(t1-t0, 2)} and average end episode length: {sum(episode_durations[-10:])/len(episode_durations[-10:])}")

    print('Complete')
    return runs_results, target_net
