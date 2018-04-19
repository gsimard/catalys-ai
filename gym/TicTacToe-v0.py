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
gamma = 1

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

        tvars = tf.trainable_variables()
        self.tvars_norm = []
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
            self.tvars_norm.append(tf.norm(var)**2)

        self.tvars_norm = tf.reduce_sum(self.tvars_norm)
        #self.tvars_norm = tf.Print(self.tvars_norm, [self.tvars_norm], message='Norm: ')
            
        # GS: Could this log lead to NaN if the responsible output is negative ?
        # What would this do on model training ???
        self.loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-10,1.0))*self.reward_holder) # + self.tvars_norm
        #self.loss = tf.Print(self.loss, [self.loss], message='Loss: ')
        
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

myAgent = agent(lr=1e-3,s_size=9,a_size=9,h_size=32768) #Load the agent.

total_episodes = 1000000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 1

GAME_BACKLOG_LENGTH = 100000
GAME_BACKLOG_SAMPLE = 16

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

def illegal_actions(state):
    return np.flatnonzero(state)

def legal_actions(state):
    return set(np.arange(0,9)).symmetric_difference(set(illegal_actions(state)))

def agent_action(sess, state, rnd=True, dbg=False):
    # What would the agent's action be ?
    a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[state]})[0]

    # Retrieve illegal actions given state and cancel their probabilities
    a_dist[illegal_actions(state)] = 0

    # Renormalize the probabily vector
    a_dist = a_dist / np.sum(a_dist)

    if dbg:
        print(a_dist)
        
    # Pick a random action
    a = np.random.choice(a_dist, p=a_dist)
    a = np.argmax(a_dist == a)

    # Disable probabilistic pick
    if not rnd:
        a = np.argmax(a_dist)
    
    return a

# Launch the tensorflow graph
sess = tf.Session()

if True:
    sess.run(init)

    # Restore previously trained model
    tf_saver = tf.train.Saver(tf.global_variables())
    #tf_saver.restore(sess, model_checkpoint)

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
                a = agent_action(sess, state)

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

    performance = -10.0
    last_performance = performance

    abort = False
    
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    # Non-zero reward moves are kept for a while so we can sample from
    # them at training time
    game_backlog = np.array([], dtype=np.float64).reshape(0,4)
        
    while i < total_episodes:
        s = env.reset()

        # For any given game, play against a varying strength AI
        #env.unwrapped.ai_strength = 0.0 if np.random.rand(1) < 1.0 else 1.0
        
        # Begin with a purely random adversary to make sure we cover
        # different moves and get all illegal moves in
        #env.unwrapped.ai_strength = 0.0

        # Next, reactivate intelligent adversary
        #env.unwrapped.ai_strength = np.random.rand(1)
        env.unwrapped.ai_strength = 0.95
        
        env.unwrapped.self_play = False

        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # What would the agent do ?
            a = agent_action(sess, s, rnd=False)

            #if np.random.rand(1) < e:
            #    a = env.action_space.sample()

            # Make move
            s1,r,d,_ = env.step(a)

            if r == -10:
                print('WHAT?')
                a = agent_action(sess, s, dbg=True)
                print(illegal_actions(s))
                print(a)
                env.render()
            
            # To begin with, a victory, loss or tie is worth the
            # same. This is an attempt at learning to obey the
            # rules, without overlearning loosing initially, which is
            # the next best reward after -10
            # Beat down the illegal moves if there were too many
            # during last round
            #if r >= -1.0:
            #    r = 0.0
            
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r

            # If game is not over and previous step() call did not
            # make the oponent's move, do it here
            if not d:
                if env.unwrapped.self_play:
                    # What would the agent do ?
                    a = agent_action(sess, -s)
                    
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
                #if r >= -1.0:
                #    r = 0.0
                    
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

                # Discount rewards for this game
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])

                # Null rewards would not affect the gradients
                ep_history = np.array(list(filter(lambda x: x[2] != 0.0, ep_history)))

                # Keep running statistics
                total_reward.append(running_reward)
                total_lenght.append(j)

                # Can we learn something from this game ?
                if len(ep_history) > 0:
                    
                    # Keep only GAME_BACKLOG_LENGTH elements
                    #game_backlog = np.concatenate((ep_history, game_backlog))
                    #game_backlog = np.delete(game_backlog, np.s_[GAME_BACKLOG_LENGTH:], 0)

                    # Is it time to train ?
                    if len(game_backlog) >= 0 and i % update_frequency == 0:

                        # Train a few times !
                        #print('TRAINING !!!')

                        for t in range(1):
                            # Sample a training minibatch from the running backlog
                            #game_sample = game_backlog[np.random.choice(game_backlog.shape[0], GAME_BACKLOG_SAMPLE, replace=False), :]
                            game_sample = ep_history

                            feed_dict = {myAgent.reward_holder: game_sample[:,2],
                                         myAgent.action_holder: game_sample[:,1],
                                         myAgent.state_in:      np.vstack(game_sample[:,0])}

                            loss = sess.run(myAgent.loss, feed_dict=feed_dict)
                            grads = sess.run(myAgent.gradients, feed_dict=feed_dict)

                            for idx,grad in enumerate(grads):
                                gradBuffer[idx] += grad

                            feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                            _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                            for ix,grad in enumerate(gradBuffer):
                                gradBuffer[ix] = grad * 0
                    
                break

        
            #Update our running tally of scores.
        if i % 250 == 0:
            print('game_backlog size:')
            print(len(game_backlog))
            print('')

            performance = np.mean(total_reward[-250:])
            print(performance)
            env.render()

            if performance > last_performance or performance > -0.1:
                tf_saver.save(sess, model_checkpoint)
                last_performance = performance
                print('checkpoint')
            #elif performance < -5:
            #    abort = True
            #    break
            
        i += 1

        if abort:
            break

