import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

import pickle
import time
# import matplotlib.pyplot as plt
# %matplotlib inline

from gym.envs.toy_text.tictactoe import TicTacToeEnv

env = gym.make('TicTacToe-v0')

# Start with a gamma of 0.0 to learn the rules, i.e.: do not penalize
# any but the last move (which is illegal). Then, increase gamma, but
# not to 1.0, because we should penalize later moves more strongly
# than early moves.
gamma = 0.5

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0

    # This is needed because if O is ending the game, its victory must
    # be learned from
    sign = np.sign(r[-1])
    
    # Do not tamper with anything if special reward -10.0 is given:
    # penalize just the illegal move, not the game
    if r[-1] == -10.0:
        return r
    
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = sign * running_add
        sign *= -1
        # Special reward means encourage everything
        if r[-1] == 2.0:
            discounted_r[t] = 1.0

    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)

        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # GS: Could this log lead to NaN if the responsible output is negative ?
        # What would this do on model training ???
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        # GS: however many states are provided FOR A SINGLE GAME, a
        # single mean scalar loss is calculated and used to calculate
        # gradients on tvars for THAT game. Then, "update_frequency"
        # games are used to calculate an average gradient before they
        # are actually applied for training. Are gradients normalized
        # for their quantities or can this increase the effective lr
        # by as much ?
        self.gradients = tf.gradients(self.loss, tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-4,s_size=9,a_size=9,h_size=32768) #Load the agent.

total_episodes = 1000000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 10

init = tf.global_variables_initializer()

e = 0.1

model_checkpoint = '/home/simard/git/tmp/model.chkpt'

def int2base(x, base):
    digits = []

    if x == 0:
        digits.append(-1)

    while x:
        digits.append(int(x % base) - 1)
        x = int(x / base)

    #digits.reverse()
        
    for i in range(9,len(digits),-1):
        digits.append(-1)
        
    return digits

def build_solution():
    ttt = gym.envs.toy_text.tictactoe.TicTacToeEnv()
    tictactoe_dict = {}
    for i in range(0, 3**9):
        print(i)
        k = np.array(int2base(i,3))
        ttt.state = k
        j,_ = ttt.minimax()
        tictactoe_dict[tuple(k)] = j
        
    with open('tictactoe_dict.pkl', 'wb') as f:
        pickle.dump(tictactoe_dict, f)
        
    return tictactoe_dict

#time.sleep(60)

def agent_action(s):
    # What would the agent's action be ?
    a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
    a = np.argmax(a_dist)
    return a

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)

    # Restore previously trained model
    tf_saver = tf.train.Saver(tf.global_variables())
    tf_saver.restore(sess, model_checkpoint)

    if False:
        # Thoroughly compare model decisions with perfect player
        res = {}
        i = 0
        j = 0
        k = 0
        for state,moves in TicTacToeEnv.tictactoe_dict.items():
            state = np.array(state)
            env.unwrapped.state = state
            env.unwrapped.first_play = 1

            # Don't bother evaluating completed games or illegal states,
            # consider only X playing
            if (not env.unwrapped.done()
                and env.unwrapped.legal()
                and env.unwrapped.whose_turn() == 1):

                # What would the agent do ?
                a = agent_action(state)

                # What would a perfect player do ?
                moves = TicTacToeEnv.tictactoe_dict[tuple(state)]

                # Record this decision and its quality
                res[tuple(state)] = (a, a in moves)

                # Increment the number of states evaluated
                j = j + 1

                # Increment the number of good decisions
                if a in moves:
                    k = k + 1

                #print('yes' if a in moves else ('no: ' + str(moves)))
                #print(str(a) + ' ' + ('X' if p==1 else 'O'))
                #env.render()

                # 3**9 = 19683
                # 6510/9040 optimal moves

            i = i + 1
            print(i)

        res_bad = {k:v for (k,v) in res.items() if not v[1]}
        res_bad_pos = [0] * 9
        for p in res_bad.values():
            res_bad_pos[p[0]] += 1

        print(res_bad_pos)

        res_good = {k:v for (k,v) in res.items() if v[1]}
        res_good_pos = [0] * 9
        for p in res_good.values():
            res_good_pos[p[0]] += 1
            
        print(res_good_pos)
           
        print(k)
        print(j)
        print(k/j)
      
        time.sleep(1000)
    
    i = 0
    total_reward = []
    total_lenght = []

    n_wrong = 0
    beatdown = False
    
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:
        s = env.reset()

        # For any given game, play against a varying strength AI
        #env.unwrapped.ai_strength = 0.0 if np.random.rand(1) < 1.0 else 1.0
        
        # Begin with a purely random adversary to make sure we cover
        # different moves and get all illegal moves in
        #env.unwrapped.ai_strength = 0.0

        # Next, reactivate intelligent adversary
        #env.unwrapped.ai_strength = np.random.rand(1)
        env.unwrapped.ai_strength = 0.5
        
        env.unwrapped.self_play = False

        #beatdown = True
        
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            # Disable probabilistic pick
            a = np.argmax(a_dist)

            #if np.random.rand(1) < e:
            #    a = env.action_space.sample()

            # Make move
            s1,r,d,_ = env.step(a)

            # To begin with, a victory, loss or tie is worth the
            # same. This is an attempt at learning to obey the
            # rules, without overlearning loosing initially, which is
            # the next best reward after -10
            # Beat down the illegal moves if there were too many
            # during last round
            if beatdown:
                if r >= -1.0:
                    r = 0.0
            
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r

            # If game is not over and previous step() call did not
            # make the oponent's move, do it here
            if not d:
                if env.unwrapped.self_play:
                    #Probabilistically pick an action given our network outputs.
                    a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[-s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # Disable probabilistic pick
                    a = np.argmax(a_dist)
                    
                    # Do play, if allowed
                    if env.unwrapped.state[a] == 0.0:
                        env.unwrapped.state[a] = env.unwrapped.whose_turn()
                        s1,r,d,_ = np.array(env.unwrapped.state), env.unwrapped.reward(), env.unwrapped.done(), {}
                    else:
                        s1,r,d,_ = np.array(env.unwrapped.state), -10.0, True, {}

                else:
                    # This is not real self-play right now
                    if np.random.rand(1) < env.unwrapped.ai_strength:
                        a = env.unwrapped.play_perfect()
                    else:
                        a = env.unwrapped.play_random()
                        
                    s1,r,d,_ = np.array(env.unwrapped.state), env.unwrapped.reward(), env.unwrapped.done(), {}

                # To begin with, a victory, loss or tie is worth the
                # same. This is an attempt at learning to obey the
                # rules, without overlearning loosing initially, which is
                # the next best reward after -10
                # Beat down the illegal moves if there were too many
                # during last round
                if beatdown:
                    if r >= -1.0:
                        r = 0.0
                    
                # Negate state to learn from other player's move: the
                # agent under training only plays X.
                ep_history.append([-s,a,r,s1])
                s = s1
                running_reward += r
            
            # To begin with, a victory, loss or tie is worth the
            # same. This is an attempt at learning to obey the
            # rules, without overlearning loosing initially, which is
            # the next best reward after -10
            #if r >= -1.0:
            #   r = 0.0
            
            # From now on, do not overstress victories
            #if r == 1.0:
            #    r = 0.1

            # From now on, penalize losses heavily
            # if r == -1.0:
            #     r = -10.0

            # Turn ties into positive rewards to learn this strategy,
            # which is the best you can do against a perfect player.
            # if d == True and r == 0.0:
            #     r = 1.0
            
            if d == True:

                # Report the lost
                # if env.unwrapped.reward() == -1:
                #     env.render()
                #     time.sleep(5)
                    
                #print('res: ' + str(r))
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                
                #print('ep_history')
                #print(ep_history)
                #time.sleep(60)
                
                feed_dict = {myAgent.reward_holder: ep_history[:,2],
                             myAgent.action_holder: ep_history[:,1],
                             myAgent.state_in:      np.vstack(ep_history[:,0])}

                #print(feed_dict)

                #print('ep_history size: ')
                #print(ep_history)
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)

                #print('grads size: ')
                #print(grads)

                #print('indexes:')
                #return
                
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                #print('')
                #time.sleep(60)
                
                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_lenght.append(j)
                break

        
            #Update our running tally of scores.
        if i % 100 == 0:
            n_wrong = len(np.where(np.array(total_reward[-100:]) == -10.0)[0])
            if n_wrong > 20:
                beatdown = True
            elif n_wrong < 10:
                beatdown = False
                
            print(str(n_wrong) + ' beatdown' if beatdown else '')
            print(np.mean(total_reward[-100:]))
            env.render()

        if i % 1000 == 0:
            tf_saver.save(sess, model_checkpoint)
            print('checkpoint')
            
        i += 1
    
    tf_saver.save(sess, model_checkpoint)
