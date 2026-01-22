# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # how many iterations
        print("we are in this runValueIteration")
        the_iteration = self.iterations
        print("the_iteration is:")
        print(the_iteration)
        # if we need no iteration we just do nothing
        if the_iteration == 0:
            return
        for i in range(the_iteration):
            new_values = self.values.copy()
            # consider all the states
            for j in self.mdp.getStates():
                if j == None:
                    return
                # this is very important and waste me a lot of time
                # once we update the terminal, it will gradually influence evey state
                if self.mdp.isTerminal(j):
                    continue
                actions = self.mdp.getPossibleActions(j)
                max_value = -1000000000
                best_action = None
                if actions == None:
                    return
                for a in actions:
                    if self.getQValue(j, a) >= max_value:
                        max_value = self.getQValue(j, a)
                        best_action = a
                new_values[j] = max_value
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # after get the action, we need to know what is the next state and the pro
        the_state = self.mdp.getTransitionStatesAndProbs(state, action)
        the_final_reward = 0
        if the_state == None:
            return
        for i in the_state:
            if i == None:
                continue
            # the next state and the probability
            next_state, pro = i
            if pro == 0:
                continue
            if next_state == None:
                return
            # print("next_state and pro, ")
            # print(next_state)
            # print(pro)
            the_reward = self.mdp.getReward(state, action, next_state)
            # Q = r + d * Q(next_state)
            the_reward = the_reward + self.discount * self.values[next_state]
            # multiply the probability and sum up all conditions
            the_final_reward += the_reward * pro
        # print(the_final_reward)
        return the_final_reward
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Note that if there are no legal actions, which is the case at the terminal state, you should return None.
        if self.mdp.getPossibleActions(state) == None:
            return None
        actions = self.mdp.getPossibleActions(state)
        if actions == None:
            return
        best_value = -100000000000
        best_action = None
        # NOTE!!!!!!!!  if two action have same value, we choose the last one
        for i in actions:
            v = self.computeQValueFromValues(state, i)
            if best_value <= v:
                best_action = i
                best_value = v
        return best_action

    # the same with above but return value instead of action
    def computeActionFromValues2(self, state):
        """
          similar above but return value
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Note that if there are no legal actions, which is the case at the terminal state, you should return None.
        if self.mdp.getPossibleActions(state) == None:
            return None
        actions = self.mdp.getPossibleActions(state)
        if actions == None:
            return
        best_value = -100000000000
        best_action = None
        # NOTE!!!!!!!!  if two action have same value, we choose the last one
        for i in actions:
            v = self.computeQValueFromValues(state, i)
            if best_value <= v:
                best_action = i
                best_value = v
        return best_value

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        the_state = self.mdp.getStates()
        if not the_state:
            return
        # we update state on by one
        for i in range(self.iterations):
            which_state = i % len(the_state)
            state_now = the_state[which_state]
            if state_now == None:
                return
            if self.mdp.isTerminal(state_now):
                continue
            else:
                # we consider all the actions
                self.values[state_now] = self.computeActionFromValues2(state_now)


def my_abs(a, b):
    value = a-b
    if value < 0:
        value = - value
    return value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)



    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # we now need to get the predecessors
        # predecessors of a state s as all states that have a nonzero probability
        # of reaching s by taking some action a.
        predecessors = {}
        for ii in self.mdp.getStates():
            if ii == None:
                return
            if self.mdp.isTerminal(ii):
                continue
            # we do the updating
            else:
                for a in self.mdp.getPossibleActions(ii):
                    for jj in self.mdp.getTransitionStatesAndProbs(ii, a):
                        next_state, pro = jj
                        if pro != 0:
                            if next_state in predecessors:
                                if predecessors[next_state] != None and predecessors[next_state] != {}:
                                    predecessors[next_state].add(ii)
                                else:
                                    predecessors[next_state] = {ii}
                            else:
                                predecessors[next_state] = {ii}
                        else:
                            continue
        # print(predecessors)
        # first we push state into the queue based on diff
        the_queue = util.PriorityQueue()
        for i in self.mdp.getStates():
            if i == None:
                return
            if self.mdp.isTerminal(i):
                continue
            else:
                the_value = self.computeActionFromValues2(i)
                diff = -my_abs(self.values[i], the_value)
                the_queue.push(i, diff)
        # we now pop from top
        for temp in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if the_queue.isEmpty():
                return
            the_state = the_queue.pop()
            if self.mdp.isTerminal(the_state):
                continue
            if the_state == None:
                break
            # we now update queue
            else:
                the_value = self.computeActionFromValues2(the_state)
                if the_state not in self.values:
                    break
                # Update the value of s (if it is not a terminal state) in self.values
                self.values[the_state] = the_value
                # similar with previous
                for i in predecessors[the_state]:
                    if self.mdp.isTerminal(i):
                        continue
                    if i == None:
                        break
                    else:
                        the_value = self.computeActionFromValues2(i)
                        diff = my_abs(self.values[i], the_value)
                        if diff > self.theta:
                            diff = -diff
                            the_queue.update(i, diff)
