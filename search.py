# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    from util import Stack
    start = problem.getStartState()  # init pos with the start state
    exploredSet = Stack()  # use it as the fringe of graph search

    visited = []  # Visited states
    path = []  # Every state keeps it's path from the starting state

    if (problem.isGoalState(start)):
        return []
    exploredSet.push((start, []))  # push start to the stack

    while (True):
        if exploredSet.isEmpty():
            return []
        pos, path = exploredSet.pop()
        visited.append(pos)  # add node to visited list
        if (problem.isGoalState(pos)):
            return path
        successors = problem.getSuccessors(pos)  # successor,action, stepCost

        if successors:
            for succ in successors:
                if succ[0] not in visited:  # if visited and set doesnt contains succ, add it
                    newPath = path + [succ[1]]
                    exploredSet.push((succ[0], newPath))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    start = problem.getStartState()  # init pos with the start state
    exploredSet = Queue()  # use it as the fringe of graph search

    visited = []  # Visited states
    path = []  # Every state keeps it's path from the starting state

    if (problem.isGoalState(start)):
        return []
    exploredSet.push((start, []))  # push start to the stack

    while (True):
        flag = False
        if exploredSet.isEmpty():
            return []
        node, path = exploredSet.pop()
        visited.append(node)  # add node to visited list
        if (problem.isGoalState(node)):
            return path

        # check if the node is not in the exploredSet
        for nodes in exploredSet.list:
            if (node in nodes[0]):
                flag = True
        if (not flag):
            successors = problem.getSuccessors(node)  # expand the node successor,action, stepCost
            if successors:
                for succ in successors:
                    if succ[0] not in visited:  # if visited and set doesnt contains succ, add it
                        newPath = path + [succ[1]]
                        exploredSet.push((succ[0], newPath))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    from util import PriorityQueue
    start = problem.getStartState()  # init pos with the start state
    exploredSet = PriorityQueue()  # use it as the fringe of graph search

    visited = []  # Visited states
    path = []  # Every state keeps it's path from the starting state

    if problem.isGoalState(start):
        return []
    # ((node, path, cost),priority)
    exploredSet.update((start, [], 0), 0)  # push start to the priority queue

    while (True):
        if exploredSet.isEmpty():
            return []
        node, path, cost = exploredSet.pop()
        visited.append(node)  # add node to explored
        if problem.isGoalState(node):
            return path
        successors = problem.getSuccessors(node)  # successor,action, stepCost

        if successors:
            for nextNode, action, newCost in successors:
                if nextNode not in visited:  # if visited and set doesnt contains succ, add it
                    newPath = path + [action]  # cal the new path associated with the current node
                    newCost += cost  # cal the cost after adding the cost of getting that node
                    exploredSet.update((nextNode, newPath, newCost), newCost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    startNode = problem.getStartState()
    expandedSet = util.PriorityQueue()
    visited = []

    if problem.isGoalState(startNode):
        return []
    expandedSet.push((startNode, [], 0), heuristic)

    while not expandedSet.isEmpty():
        node, path, cost = expandedSet.pop()  # pop the node based on priority

        if node not in visited:
            visited.append(node)
            if problem.isGoalState(node):  # node is not the goal
                return path
            successors = problem.getSuccessors(node)
            for nextNode, newPath, newCost in successors:
                if nextNode not in visited:
                    newCost += cost
                    newPath = path + [newPath]
                    heuristicCost = newCost + heuristic(nextNode, problem)
                    expandedSet.push((nextNode, newPath, newCost), heuristicCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
