## ID: 20212593 NAME: Jeon, Yejin
######################################################################################
# Problem 2a
# minimax value of the root node: 5
# pruned edges: h, m
######################################################################################

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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newscared_time holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newscared_time = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    def min_value(gameState,agentnumber,current_depth):
      next_agent = agentnumber + 1
      value = (float("inf"), Directions.STOP)

      if next_agent == gameState.getNumAgents():
        next_agent = 0
        current_depth = current_depth - 1
      for action in gameState.getLegalActions(agentnumber): #for each successor of state
        next_state = gameState.generateSuccessor(agentnumber,action)
        new_value = minimax_value(next_state,next_agent,current_depth)

        if new_value < value[0]:
          value=(new_value,action)

      return value

    def max_value(gameState,agentnumber,current_depth):
      next_agent = agentnumber + 1
      value=(float("-inf"),Directions.STOP)

      for action in gameState.getLegalActions(agentnumber): #for each successor of state
        next_state = gameState.generateSuccessor(agentnumber,action)
        new_value = minimax_value(next_state,next_agent,current_depth)

        if new_value > value[0]:
          value = (new_value,action)

      return value

    def minimax_value(gameState,agentnumber,current_depth):
      if gameState.isLose() or gameState.isWin():
        return gameState.getScore()

      if current_depth <= 0:
        return self.evaluationFunction(gameState)
        
      if agentnumber == 0:
        return max_value(gameState,agentnumber,current_depth)[0]
      else:
        return min_value(gameState,agentnumber,current_depth)[0]

    return max_value(gameState,0,self.depth)[1]
    # END_YOUR_ANSWER

######################################################################################
# Problem 2b: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER (our solution is 42 lines of code, but don't worry if you deviate from this)
    def min_value(gameState,agentnumber,current_depth,alpha,beta):
      
      next_agent = agentnumber + 1
      value=(float("inf"),Directions.STOP)
  
      if next_agent == gameState.getNumAgents():
        next_agent=0
        current_depth = current_depth -1

      for action in gameState.getLegalActions(agentnumber): #for each successor of state
        next_state = gameState.generateSuccessor(agentnumber,action)
        new_value = minimax_value(next_state,next_agent,current_depth,alpha,beta)
        # pruning
        if new_value < value[0]:
          value = (new_value,action)
        beta = min(beta,value[0])
        
        if value[0] < alpha:
          return value

      return value

    def max_value(gameState,agentnumber,current_depth,alpha,beta):

      next_agent = agentnumber+1
      value = (float("-inf"),Directions.STOP)
      
      for action in gameState.getLegalActions(agentnumber): #for each successor of state
        next_state=gameState.generateSuccessor(agentnumber,action)
        new_value=minimax_value(next_state,next_agent,current_depth,alpha,beta)
        # pruning
        if new_value > value[0]: 
          value = (new_value,action)
        alpha = max(alpha,value[0])
        
        if value[0] > beta:
          return value

      return value


    def minimax_value(gameState,agentnumber,current_depth,alpha,beta):
      if gameState.isLose() or gameState.isWin():
        return gameState.getScore()

      if current_depth <= 0:
        return self.evaluationFunction(gameState)

      if agentnumber == 0:
        return max_value(gameState, agentnumber, current_depth,alpha,beta)[0]
      else:
        return min_value(gameState,agentnumber,current_depth,alpha,beta)[0]

    return max_value(gameState,0,self.depth,float("-inf"),float("inf"))[1]
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    
    def expectimax_value(gameState,agentnumber,current_depth):
     
      next_agent = agentnumber + 1
      value = [0.0]
    
      if next_agent == gameState.getNumAgents():
        next_agent = 0
        current_depth = current_depth- 1

      for action in gameState.getLegalActions(agentnumber):
        next_state = gameState.generateSuccessor(agentnumber,action)
        new_value = minimax_value(next_state,next_agent,current_depth)
        value.append(new_value)

      value.pop(0)

      return sum(value)/len(value)

    def max_value(gameState,agentnumber,current_depth):
      
      next_agent = agentnumber + 1
      value = (float("-inf"),Directions.STOP)
      
      for action in gameState.getLegalActions(agentnumber):
        next_state = gameState.generateSuccessor(agentnumber,action)
        new_value = minimax_value(next_state,next_agent,current_depth)

        if new_value > value[0]:
          value = (new_value,action)

      return value

    def minimax_value(gameState,agentnumber,current_depth):
      if gameState.isLose() or gameState.isWin():
        return gameState.getScore()

      if current_depth <= 0:
        return self.evaluationFunction(gameState)

      if agentnumber == 0:
        return max_value(gameState,agentnumber,current_depth)[0]
      else:
        return expectimax_value(gameState,agentnumber,current_depth)

    return max_value(gameState,0,self.depth)[1]
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 4).
  """

  # BEGIN_YOUR_ANSWER (our solution is 60 lines of code, but don't worry if you deviate from this)
  ghost_states = currentGameState.getGhostStates()
  food = currentGameState.getFood()
  food_asList = food.asList()
  

  closest_food = float("inf")
  for food in food_asList:
    distance = util.manhattanDistance(currentGameState.getPacmanPosition(), food)
    if closest_food > distance:
      closest_food = distance

  hunt_ghosts = []
  scared_ghosts = []
  scared_time = []

  for ghost_agent in ghost_states:
    if ghost_agent.scaredTimer:
      scared_ghosts.append(ghost_agent)
      scared_time.append(ghost_agent.scaredTimer)
    else:
      hunt_ghosts.append(ghost_agent)

     
  distance_hunt_ghost = float("inf")
  for ghost_agent in hunt_ghosts:
    distance = util.manhattanDistance(currentGameState.getPacmanPosition(), ghost_agent.getPosition())
    if distance_hunt_ghost > distance:
      distance_hunt_ghost = distance

  if len(scared_ghosts) == 0:
    distance_scared_ghost = 0
    scared_time=0
  else:
    distance_scared_ghost = float("inf")
    for ghost_agent in scared_ghosts:
      distance = util.manhattanDistance(currentGameState.getPacmanPosition(), ghost_agent.getPosition())
      
      if distance_scared_ghost > distance:
        distance_scared_ghost = distance
    scared_time = scared_time[0]
    
  inverse_hunt_ghosts = 0
  if distance_hunt_ghost > 0:
    inverse_hunt_ghosts = (float)(1) / distance_hunt_ghost
  invers_scared_ghosts = 0
  if distance_scared_ghost > 0:
    invers_scared_ghosts = (float)(1) / distance_scared_ghost


  inverse_capsule = 0
  closest_capsule = float("inf")
 
  capsule=currentGameState.getCapsules()
  if len(capsule) == 0:
    closest_capsule = 0

  for bigfood in capsule:
    distance = util.manhattanDistance(currentGameState.getPacmanPosition(), bigfood)
    if distance < closest_capsule:
      closest_capsule = distance

  if closest_capsule > 0:
    inverse_capsule= 1/ (closest_capsule +10000)
    
  score_scared = 10*scared_time*invers_scared_ghosts
  score_hunt = 2*inverse_hunt_ghosts
  score_food = 2*len(food_asList)
  score_capsule = 5*inverse_capsule

  return currentGameState.getScore() + score_scared -score_hunt -score_food -closest_food -score_capsule
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

