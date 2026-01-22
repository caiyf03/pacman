# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        cap=childGameState.getCapsules()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghost_dis=999
        ghost_position = childGameState.getGhostPositions()
        for ghost_po in ghost_position:
            ghost_distance = manhattanDistance(ghost_po,newPos)
            if(ghost_distance < 2): # ghost can eat it we never choose it
                return -999
            if(ghost_distance<ghost_dis):
                ghost_dis = ghost_distance

        food_dis=999
        food = newFood.asList()
        if food:
            for food_temp in food:
                food_distance = manhattanDistance(food_temp,newPos)
                if(food_distance < food_dis):
                    food_dis = food_distance
        else:
            food_dis = 0

        c_dis = 999
        if cap:
            for food_temp in cap:
                food_distance = manhattanDistance(food_temp, newPos)
                if (food_distance < c_dis):
                    c_dis = food_distance
        else:
            c_dis = 0


        if newScaredTimes ==0:
            if c_dis == 0:
                score=999

            if food_dis == 0:  # we eat
                if ghost_dis > 5 and ghost_dis < 9:
                    score = childGameState.getScore() + 20 + ghost_dis / 10
                elif ghost_dis >=9:
                    core = childGameState.getScore() + 20 + ghost_dis / 10
                else:
                    score = childGameState.getScore() + 20 + ghost_dis ** 2
            else:
                if ghost_dis > 5 and ghost_dis <9:
                    score = childGameState.getScore() + 3 / food_dis + ghost_dis / 10
                elif ghost_dis >= 9:
                    score = childGameState.getScore() + 3 / food_dis + ghost_dis / 10
                else:
                    score = childGameState.getScore() + 3 / food_dis + ghost_dis ** 2
        else:
            if food_dis == 0:  # we eat
                score = childGameState.getScore() + 20
            else:
                score = childGameState.getScore() + 3 / food_dis

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def is_pacman(self, gameState, agent, depth):
        best_value = -999
        for action in gameState.getLegalActions(agent):
            child = gameState.getNextState(agent, action)
            v = self.minimax(child, agent + 1, depth)
            if v >= best_value:
                best_value = v
                if depth == 1:
                    self.action = action
            else:
                best_value = best_value
        return best_value

    def is_ghost(self, gameState, agent, depth):
        best_value = 999
        for action in gameState.getLegalActions(agent):
            successor = gameState.getNextState(agent, action)
            v = self.minimax(successor, agent + 1, depth)
            if v < best_value:
                best_value = v
            else:
                best_value = best_value
        return best_value

    def minimax(self, gameState, agent=0, depth=0):
        agent = agent % gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and agent==0):
            return self.evaluationFunction(gameState)
        else:
            if agent == 0:  # whether is pacman we need to add the depth
                flag = 0
            else:
                flag = 1
        if flag == 0:
            return self.is_pacman(gameState, agent, depth + 1)
        else:
            return self.is_ghost(gameState, agent, depth)
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.minimax(gameState,0)
        return self.action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.ab_purn(gameState,0)
        return self.action

    def is_pacman(self, gameState, agent, depth, alpha, beta):
        best_value = -9999
        for action in gameState.getLegalActions(agent):
            child = gameState.getNextState(agent, action)
            v = self.ab_purn(child, agent + 1, depth, alpha, beta)
            if v >= best_value:
                best_value = v
                if depth == 1:
                    self.action = action
                else:
                    print(depth)
            else:
                best_value = best_value
            if best_value > beta:
                return best_value
            if alpha <= best_value:
                alpha = best_value
        return best_value

    def is_ghost(self, gameState, agent=0, depth=0, alpha=-9999, beta=9999):
        best_value = 9999
        for action in gameState.getLegalActions(agent):
            successor = gameState.getNextState(agent, action)
            v = self.ab_purn(successor, agent + 1, depth, alpha, beta)
            if v < best_value:
                best_value = v
            else:
                best_value = best_value

            if best_value < alpha:
                return best_value
            if beta >= best_value:
                beta = best_value
        return best_value

    def ab_purn(self, gameState, agent, depth=0, alpha=-9999,beta=9999):
        agent = agent % gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and agent == 0):
            return self.evaluationFunction(gameState)
        else:
            if agent == 0:  # whether is pacman we need to add the depth
                flag = 0
            else:
                flag = 1
        if flag == 0:
            return self.is_pacman(gameState, agent, depth + 1, alpha, beta)
        else:
            return self.is_ghost(gameState, agent, depth, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def is_pacman(self, gameState, agent, depth):
        best_value = -9999
        count = 0
        for action in gameState.getLegalActions(agent):
            count = count + 1
            child = gameState.getNextState(agent, action)
            v = self.naive_ghost(child, agent + 1, depth)
            if v >= best_value:
                best_value = v
                if depth == 1:
                    self.action = action
            else:
                best_value = best_value
            #if best_value >beta:
                #return best_value
            #if alpha <= best_value:
               # alpha = best_value
        return best_value

    def is_ghost(self, gameState, agent, depth):
        best_value = 0
        count = 0
        for action in gameState.getLegalActions(agent):
            count= count + 1
            successor = gameState.getNextState(agent, action)
            best_value = self.naive_ghost(successor, agent + 1, depth)+best_value
        best_value=best_value/len(gameState.getLegalActions(agent))
            #if bestValue < alpha:
              #  return bestValue
            #if beta >= bestValue:
               # beta = bestValue
        return best_value

    def naive_ghost(self, gameState, agent=0, depth=0,alpha=-9999,beta=9999):
        agent = agent % gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and agent == 0):
            return self.evaluationFunction(gameState)
        else:
            if agent == 0:  # whether is pacman we need to add the depth
                flag = 0
            else:
                flag = 1
        if flag == 0:
            return self.is_pacman(gameState, agent, depth + 1)
        else:
            return self.is_ghost(gameState, agent, depth)
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.naive_ghost(gameState,0)
        return self.action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    pac_position = currentGameState.getPacmanPosition()
    scoree = currentGameState.getScore()
    fsscore=0
    ghost_state = currentGameState.getGhostStates()
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghost_dis = 9999
    if ghost_state:
        for ghost in ghost_state:
            ghost_position = ghost.getPosition()
            diss = manhattanDistance(ghost_position,pac_position)
            if ghost_dis > diss:
                ghost_dis = diss
            else:
                ghost_dis = ghost_dis
            if ghost.scaredTimer > 6 and ghost_dis < 2:
                ghost_dis = 10 + 1/ghost_dis
    else:
        ghost_dis = 0

    food_dis = 9999
    if foods.asList():
        for food in foods.asList():
            diss = manhattanDistance(food,pac_position)
            if food_dis > diss:
                food_dis = diss
    else:
        food_dis = 0

    capsule_dis = 9999
    if capsules:
        for capsule in capsules:
            diss = manhattanDistance(capsule, pac_position)
            if capsule_dis > diss:
                capsule_dis = diss
    else:
        capsule_dis = 0.5

    if food_dis == 0:  # we eat
        if ghost_dis >5:
            fsscore = scoree + 10 + ghost_dis / 10 + 10/capsule_dis
        else:
            fsscore = scoree + 10 + ghost_dis + 10 / capsule_dis
    else:
        if ghost_dis > 5:
            fsscore = scoree + 8 / food_dis + ghost_dis / 10 + 10/capsule_dis
        else:
            fsscore = scoree + 8 / food_dis + ghost_dis + 10 / capsule_dis

    return fsscore

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    """
              Returns an action.  You can use any method you want and search to any depth you want.
              Just remember that the mini-contest is timed, so you have to trade off speed and computation.

              Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
              just make a beeline straight towards Pacman (or away from him if they're scared!)
            """
    "*** YOUR CODE HERE ***"
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.ab_purn(gameState,0)
        return self.action

    def is_pacman(self, gameState, agent, depth, alpha, beta):
        best_value = -9999
        for action in gameState.getLegalActions(agent):
            child = gameState.getNextState(agent, action)
            v = self.ab_purn(child, agent + 1, depth, alpha, beta)
            if v >= best_value:
                best_value = v
                if depth == 1:
                    self.action = action
                else:
                    print(depth)
            if best_value > beta:
                return best_value
            if alpha <= best_value:
                alpha = best_value
        return best_value

    def is_ghost(self, gameState, agent=0, depth=0, alpha=-9999, beta=9999):
        best_value = 9999
        for action in gameState.getLegalActions(agent):
            successor = gameState.getNextState(agent, action)
            v = self.ab_purn(successor, agent + 1, depth, alpha, beta)
            if v < best_value:
                best_value = v
            if best_value < alpha:
                return best_value
            if beta >= best_value:
                beta = best_value
        return best_value

    def ab_purn(self, gameState, agent, depth=0, alpha=-9999,beta=9999):
        agent = agent % gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and agent == 0):
            return betterEvaluationFunction(gameState)
        if agent == 0:
            return self.is_pacman(gameState, agent, depth + 1, alpha, beta)
        else:
            return self.is_ghost(gameState, agent, depth, alpha, beta)

        #return AlphaBetaAgent().getAction(gameState)
        #return MinimaxAgent().getAction(gameState)

        #util.raiseNotDefined()