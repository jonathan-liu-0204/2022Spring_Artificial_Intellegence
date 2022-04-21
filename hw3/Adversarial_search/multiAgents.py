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

        # get all the legal actions of pacman
        pacman_legal_actions = gameState.getLegalActions(0)

        print()
        print(pacman_legal_actions)
        
        max_value = float('-inf')
        max_action  = None #one to be returned at the end.

        for action in pacman_legal_actions:   #get the max value from all of it's successors.
            action_value = self.Min_Value(gameState.getNextState(0, action), 1, 0)
            print("action: ", action, "action_value: ", action_value)
            if ((action_value) > max_value ): #take the max of all the children.
                max_value = action_value
                max_action = action

        print("final_call", max_action)
        return max_action #Returns the final action

        util.raiseNotDefined()  
        # End your code
    
    def Max_Value (self, gameState, depth):
        """For the Max Player here Pacman"""

        if ((depth == self.depth)  or (len(gameState.getLegalActions(0)) == 0)):
            return self.evaluationFunction(gameState)

        print("Max_Value Self Depth: ", self.depth)

        return max([self.Min_Value(gameState.getNextState(0, action), 1, depth) for action in gameState.getLegalActions(0)])


    def Min_Value (self, gameState, agentIndex, depth):
        """ For the MIN Players or Agents  """

        if (len(gameState.getLegalActions(agentIndex)) == 0): #No Legal actions.
            return self.evaluationFunction(gameState)

        if (agentIndex < gameState.getNumAgents() - 1):
            return min([self.Min_Value(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) for action in gameState.getLegalActions(agentIndex)])

        else:  #the last ghost HERE
            return min([self.Max_Value(gameState.getNextState(agentIndex, action), depth + 1) for action in gameState.getLegalActions(agentIndex)])

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
        
        pacman_legal_actions = gameState.getLegalActions(0) #all the legal actions of pacman.
        max_value = float('-inf')
        max_action  = None #one to be returned at the end.

        for action in pacman_legal_actions:   #get the max value from all of it's successors.
            action_value = self.Min_Value(gameState.getNextState(0, action), 1, 0)
            if ((action_value) > max_value ): #take the max of all the children.
                max_value = action_value
                max_action = action

        return max_action #Returns the final action .8

        # End your code
    
    def Max_Value (self, gameState, depth):
        """For the Max Player here Pacman"""

        if ((depth == self.depth)  or (len(gameState.getLegalActions(0)) == 0)):
            return self.evaluationFunction(gameState)

        return max([self.Min_Value(gameState.getNextState(0, action), 1, depth) for action in gameState.getLegalActions(0)])


    def Min_Value (self, gameState, agentIndex, depth):
        """ For the MIN Players or Agents  """

        num_actions = len(gameState.getLegalActions(agentIndex))

        if (num_actions == 0): #No Legal actions.
            return self.evaluationFunction(gameState)

        if (agentIndex < gameState.getNumAgents() - 1):
            return sum([self.Min_Value(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) for action in gameState.getLegalActions(agentIndex)]) / float(num_actions)

        else:  #the last ghost HERE
            return sum([self.Max_Value(gameState.getNextState(agentIndex, action), depth + 1) for action in gameState.getLegalActions(agentIndex)]) / float(num_actions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (part1-3).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Begin your code

    # Setup information to be used as arguments in evaluation function
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

    # Find distances from pacman to all food
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]

    # Set value for closest food if there is still food left
    if food_count > 0:
        closest_food = min(food_distances)

    # Find distances from pacman to ghost(s)
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghost_distance < 2:
            closest_food = 99999

    features = [1.0 / closest_food, game_score, food_count, capsule_count]

    weights = [10, 200, -100, -10]

    # Linear combination of features
    return sum([feature * weight for feature, weight in zip(features, weights)])

    util.raiseNotDefined()
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = scoreEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
