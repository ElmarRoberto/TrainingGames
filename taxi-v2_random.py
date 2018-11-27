import gym

env = gym.make('Taxi-v2') # load the environment

# print('State space', env.observation_space.n) # Amount of available states
# print('Action space', env.action_space.n) # Amount of available actions
# values = [new_state, reward, is_done, info_for_debug]
# messages = ['New state:', 'Reward of the current state:',
#             'Has the agent finished the task ?', 'Info for debugging:']

number_episodes = 100
total_train = 10

open('random_text_results.txt', 'w').close()
file = open('random_text_results.txt', 'a')

for aux in range(total_train):
  total_loss = 0
  total_wins = 0
  total_state_remained = []
  for i in range(number_episodes):
    # Reset the environment to initialize a new ep
    initial_state_of_ep = env.reset()

    #print('\n========== EPISODE START ==========\n')
    #print(f'The initial episode state are: {initial_state_of_ep}')

    is_done = False
    loss, wins, state_remained = 0, 0, 0
    current_state = initial_state_of_ep

    while not is_done:
      # env.render() -> use this command to see the game runing

      # To see the current state
      #print(f'\nState BEFORE the action: {current_state}')

      random_action = env.action_space.sample() # Take a random action
      new_state, reward, is_done, info_for_debug = env.step(random_action)

      #print(f'State AFTER the action: {new_state}')

      # The state remained
      if current_state == new_state:
        state_remained += 1
      
      # Loss 10 points if the car is empty and perform
      # an action involving the passenger
      if reward == -10:
        loss += 10

      if reward == 20:
        #print('WIN')
        wins += 1

      # To see informations about each action taken
      # for v, m in zip(values, messages):
      #   print('{} {}'.format(m, v));
      # print('\n')
      
      loss -= 1
      current_state = new_state

    total_loss += loss
    total_wins += wins
    total_state_remained.append(state_remained)


    # print('\n========== EPISODE FINISH ==========\n')
    # print(f'Total loss of this episode {loss}')
    # print(f'The agent won the game {wins} times in this episode')

  file.write(f'========== TRAINING {aux + 1} ENDED ==========\n')
  file.write('20000 steps were taken to complete the training\n')
  file.write(f'The agent won the game {total_wins} times\n')
  file.write(f'Average wins per episode: {total_wins / number_episodes}\n')
  file.write(f'{total_loss} Total loss until the end of training\n')
  file.write(f'Average loss per episode: {total_loss / number_episodes}\n')
  file.write(f'List of state repeats per episode [1-100] - max(200)\n {total_state_remained}\n\n')

file.close()