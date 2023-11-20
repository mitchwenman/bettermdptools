"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Model-based learning algorithms: Value Iteration and Policy Iteration

Assumes prior knowledge of the type of reward available to the agent
for iterating to an optimal policy and reward value for a given MDP.
"""

import numpy as np
import warnings
from utils.decorators import print_runtime
import time


class Planner:
    def __init__(self, P):
        self.P = P

    @print_runtime
    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        Q_track = np.zeros((n_iters, len(self.P), len(self.P[0])), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            if i % 50 == 0:
                print(f'Value iteration {i}')
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            print(f'max diff: {np.max(np.abs(V - np.max(Q, axis=1)))}')
            V = np.max(Q, axis=1)
            Q_track[i] = Q
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check theta and n_iters.  ")
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return V, Q_track, pi, Q, i

    @print_runtime
    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        import copy
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        Q_track = np.zeros((n_iters, len(self.P), len(self.P[0])), dtype=np.float64)
        i = 0
        pol_eval_time = 0
        pol_improv_time = 0
        pol_eval_iters = []
        converged = False
        while i < n_iters-1 and not converged:
            print(f'Policy iteration {i}')
            i += 1
            old_pi = copy.deepcopy(pi)
            start = time.time()
            V, conv_iters = self.policy_evaluation(pi, V, gamma, theta)
            pol_eval_iters.append(conv_iters)
            pol_eval_time += (time.time() - start)
            start = time.time()
            pi, Q = self.policy_improvement(V, gamma)
            pol_improv_time += (time.time() - start)
            Q_track[i] = Q
            if old_pi == pi:
                converged = True
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        print(f'Policy evaluation time: {pol_eval_time}')
        print(f'Policy improvement time: {pol_improv_time}')
        print(f'Policy evaluation iterations avg: {np.mean(pol_eval_iters)}')
        pi_func = lambda s: pi[s]
        return V, Q_track, pi_func, Q, i + 1

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        i = 0
        while True:
            i += 1
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        print(f'Policy evaluation converged in {i} iterations')
        return V, i

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        # Explanation of lambda:
        # def new_pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        new_pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return new_pi, Q
