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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
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
        # Begin your code

        pacman_legal_actions = gameState.getLegalActions(0)

        # print()
        # print(pacman_legal_actions)
        
        max_value = float('-inf')
        decision  = None

        for action in pacman_legal_actions:

            action_score = self.Min_Value(gameState.getNextState(0, action), 1, 0)
            # print("action: ", action, "action_score: ", action_score)

            if ((action_score) > max_value ):
                max_value = action_score
                decision = action

        # print("final_call", decision)
        return decision

        util.raiseNotDefined()  
        # End your code
    
    def Max_Value (self, gameState, depth):

        # end of the game
        if(depth == self.depth):
            return self.evaluationFunction(gameState)
        # dead
        elif(len(gameState.getLegalActions(0)) == 0): 
            return self.evaluationFunction(gameState)

        # print("Max_Value Self Depth: ", self.depth)

        return max([self.Min_Value(gameState.getNextState(0, action), 1, depth) 
                        for action in gameState.getLegalActions(0)])


    def Min_Value (self, gameState, agentIndex, depth):

        # dead
        if(len(gameState.getLegalActions(agentIndex)) == 0):
            return self.evaluationFunction(gameState)

        # when the game continues
        if(agentIndex < gameState.getNumAgents() - 1): 
            return min([self.Min_Value(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) 
                            for action in gameState.getLegalActions(agentIndex)])
        # when the last ghost remaining
        else: 
            return min([self.Max_Value(gameState.getNextState(agentIndex, action), depth + 1) 
                            for action in gameState.getLegalActions(agentIndex)])

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        
        pacman_legal_actions = gameState.getLegalActions(0)

        # print()
        # print(pacman_legal_actions)
        
        max_value = float('-inf')
        decision  = None

        for action in pacman_legal_actions:

            action_score = self.Exp_Value(gameState.getNextState(0, action), 1, 0)
            # print("action: ", action, "action_score: ", action_score)

            if ((action_score) > max_value ):
                max_value = action_score
                decision = action

        # print("final_call", decision)
        return decision

        util.raiseNotDefined()  
        # End your code
    
    def Max_Value (self, gameState, depth):

        # end of the game
        if(depth == self.depth):
            return self.evaluationFunction(gameState)
        # dead
        elif(len(gameState.getLegalActions(0)) == 0): 
            return self.evaluationFunction(gameState)

        # print("Max_Value Self Depth: ", self.depth)

        return max([self.Exp_Value(gameState.getNextState(0, action), 1, depth)
                        for action in gameState.getLegalActions(0)])


    def Exp_Value (self, gameState, agentIndex, depth):

        num_actions = len(gameState.getLegalActions(agentIndex))

        # dead
        if(len(gameState.getLegalActions(agentIndex)) == 0):
            return self.evaluationFunction(gameState)

        # when the game continues
        if(agentIndex < gameState.getNumAgents() - 1): 
            return sum([self.Exp_Value(gameState.getNextState(agentIndex, action), agentIndex + 1, depth)
                            for action in gameState.getLegalActions(agentIndex)]) / float(num_actions)
        # when the last ghost remaining
        else: 
            return sum([self.Max_Value(gameState.getNextState(agentIndex, action), depth + 1)
                            for action in gameState.getLegalActions(agentIndex)]) / float(num_actions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (part1-3).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Begin your code

    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    #number of remaining capsule
    capsule_num = len(currentGameState.getCapsules())

    # print()
    # print(currentGameState.getCapsules())

    game_score = currentGameState.getScore()

    # ------ food area ------ 
    # get the position of all food
    food_place = currentGameState.getFood()
    food_place = food_place.asList()

    #number of remaining food
    food_num = len(food_place)

    # print()
    # print(food_place)

    closest_food = 1
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_place]
    # print("food_distances: ", food_distances)

    if food_num > 0:
        closest_food = min(food_distances)

    #distance of ghost to us
    for ghost_position in ghost_positions:

        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to us, then its time to stop eating and escape lol
        if ghost_distance < 5:
            closest_food = 878787

    #calculate all the weighted features and sum them up
    features = [(1.0 / closest_food), game_score, food_num, capsule_num]
    weights = [10, 200, -100, -10]
    return sum([feature * weight for feature, weight in zip(features, weights)])

    util.raiseNotDefined()
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = betterEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
