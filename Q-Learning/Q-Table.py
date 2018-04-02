import gym
import numpy as np

from gym.envs.registration import register

env = gym.make('FrozenLakeNoSlip-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 5000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    # print("initial state: " + str(s))
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # Early-on, select random action often
        if i < 0:
            a = np.random.randint(env.action_space.n)
        else:
            #Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(int(i)+1)))
        # a = 1
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        # print("before: " + str(s))
        # print("after: " + str(s1))
        # break
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    jList.append(j)
    rList.append(rAll)

# Produce statistics on last 100 games
# % of success
fh = open("lake.out", "w")
print(rList[-100:], file=fh)
print(np.sum(rList[-100:]), file=fh)
# Average game length
print(jList[-100:], file=fh)
print(np.mean(jList[-100:]), file=fh)
fh.close()

print("Percent of succesful episodes: " + str(np.sum(rList[-2500:])/2500*100) + "%")

print("Final Q-Table Values")
print(Q)

