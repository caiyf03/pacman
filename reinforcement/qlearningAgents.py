# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from pacman import GameState
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()  # a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        the_key = (state, action)
        if the_key in self.values:
            if self.values[(state, action)] == None:
                return 0
            else:
                return self.values[(state, action)]
        else:
            return 0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # similar with valueiterationAgent
        if len(self.getLegalActions(state)) == 0 or not self.getLegalActions(state):
            return 0
        else:
            max_value = -1000000000
            the_action = []
            if len(self.getLegalActions(state)) == 0:
                print("empty")
                return
            if self.getLegalActions(state) == None:
                return
            # we now try to find the max value
            for a in self.getLegalActions(state):
                temp = self.getQValue(state, a)
                # print("temp is", temp)
                if temp > max_value:
                    max_value = temp
                    the_action = []
                elif temp == max_value:
                    max_value = temp
                    the_action.append(a)
            # print(max_value)
            return max_value
    def computeValueFromQValues1(self, state):
        """
          similar above but return action list
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # similar with valueiterationAgent
        if len(self.getLegalActions(state)) == 0 or not self.getLegalActions(state):
            return 0
        else:
            max_value = -1000000000
            the_action = []
            if len(self.getLegalActions(state)) == 0:
                print("empty")
                return
            if self.getLegalActions(state) == None:
                return
            # we now try to find the max action
            for a in self.getLegalActions(state):
                temp = self.getQValue(state, a)
                if temp > max_value:
                    max_value = temp
                    the_action = []
                    the_action.append(a)
                elif temp == max_value:
                    max_value = temp
                    the_action.append(a)
            return the_action
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # we use the above function
        if len(self.getLegalActions(state)) == 0 or not self.getLegalActions(state):
            return None
        else:
            temp = self.computeValueFromQValues1(state)
            return random.choice(temp)
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Complete your Q-learning agent by implementing epsilon-greedy action selection in getAction,
        # meaning it chooses random actions an epsilon fraction of the time,
        # and follows its current best Q-values otherwise.
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions) == 0:
            return None
        # e-greedy
        flag = util.flipCoin(self.epsilon)
        if flag == True:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # similar with valueIterationAgent
        the_reward = reward + self.discount * self.computeValueFromQValues(nextState)
        result = (1 - self.alpha) * self.values[(state, action)] + self.alpha * the_reward
        self.values[(state, action)] = result

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        Q_val = 0
        weight = 0
        # something like IdentityExtractor class imported by util.lookup
        temp = self.featExtractor.getFeatures(state, action)
        # 键值对元组列表
        for i in temp.items():
            feature, value = i
            if feature not in self.getWeights():
                # print(feature)
                # print(self.getWeights())
                self.getWeights().update({feature: 0})
                # this should never be achieved but where is my bug??
                if feature not in self.getWeights():
                    print("it can never be")
                    return
                else:
                    weight = 0
                    if self.getWeights()[feature] != 0:
                        print("what happens???")
                        return
            else:
                weight = self.getWeights()[feature]
            #  Q(state,action) = w * featureVector
            Q_val += weight * value
        return Q_val

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # based on the formular on BB
        gamma = self.discount
        Q_new = self.computeValueFromQValues(nextState)
        Q = self.getQValue(state, action)
        # update of difference
        diff = (reward + gamma * Q_new) - Q
        features = self.featExtractor.getFeatures(state, action)
        # update of weight
        for i in features.items():
            feature, value = i
            # if we can not find the key (feature), but this should never happen
            if feature not in self.getWeights():
                print("wrong?")
                continue
            # updating
            else:
                new_value = self.getWeights()[feature] + self.alpha * diff * value
                self.getWeights()[feature] = new_value


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # what's this ?
            "*** YOUR CODE HERE ***"
            print(" the final function will be run now")
            pass


class BetterExtractor(FeatureExtractor):
    "Your extractor entry goes here.  Add features for capsuleClassic."

    def getFeatures(self, state: GameState, action):
        # features = SimpleExtractor().getFeatures(state, action)
        # Add more features here
        "*** YOUR CODE HERE ***"
        # features = SimpleExtractor().getFeatures(state, action)
        # Add more features here
        "*** YOUR CODE HERE ***"
        walls = state.getWalls()
        the_foods = state.getFood()
        the_ghosts = state.getGhostPositions()
        ghostState = state.getGhostStates()
        the_capsules = state.getCapsules()
        the_pac = state.getPacmanPosition()
        features = util.Counter()
        features.update({"walls" : walls})
        features.update({"the_foods" : the_foods})
        features.update({"the_ghosts" : the_ghosts})
        min_food_dis = 1000000000
        for food in the_foods:
            min_food_dis = min(manhattanDistance(the_pac, food), min_food_dis)
        min_ghost_dis = 100000000
        for ghost in the_ghosts:
            min_ghost_dis = min(manhattanDistance(the_pac, ghost), min_ghost_dis)
        features.update({"the_ghost_dis": min_food_dis})
        features.update({"the_foods_dis": min_ghost_dis})
        print(features)

