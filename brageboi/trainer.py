from copy import deepcopy
import matplotlib.pyplot as plt
from Boards.diamondBoard import DiamondBoard
from Boards.triangleBoard import TriangleBoard
from actor import Actor
from critic import Critic
from parameters import Parameters as p
from parameters import DiamondBoard as db
from parameters import TriangleBoard as tb
from boardAnimator import BoardAnimator


class Trainer:

    def __init__(self, useDiamondBoard, showFinalSolution):
        self.showFinalSolution = showFinalSolution
        self.actor = Actor(p["actorLearningRate"], p["actorDiscFactor"], p["actorTraceDecay"])
        self.critic = Critic(p["criticLearningRate"], p["criticDiscFactor"], p["criticTraceDecay"])
        self.epsilon = p["epsilon"]
        self.epsilonDecay = p["epsilonDecay"]
        self.numberOfEpisodes = p["numberOfEpisodes"]
        if useDiamondBoard:
            self.board = DiamondBoard(db["emptyStartPins"], db["size"])
            self.boardSize = db["size"]
        else:
            self.board = TriangleBoard(tb["emptyStartPins"], tb["size"])
            self.boardSize = tb["size"]
        self.useDiamondBoard = useDiamondBoard

    def runEpisode(self):
        currentEpisode = []
        self.board.resetBoard()
        self.critic.reset()  # e(s) <- 0 all s
        self.actor.reset()  # e(s,a) <- 0 all s,a
        state = self.board.getCurrentStringState()  # s <-s_init
        self.actor.setETrace(state)
        self.critic.setETrace(state)
        self.actor.setPolicy(state)
        self.critic.setExpValues(state)
        action = self.actor.getNextAction(self.board.getLegalChildStates(), epsilon=self.epsilon)  # a <- pi(s_init)
        while self.board.legalMoveLeft():
            self.board.executeStringState(action)
            currentEpisode.append(state)
            stateTot = self.board.getCurrentStringState()  # sTot <- s on a
            self.actor.setETrace(stateTot)
            self.critic.setETrace(stateTot)
            self.actor.setPolicy(stateTot)
            self.critic.setExpValues(stateTot)
            nextStates = self.board.getLegalChildStates()
            if self.board.legalMoveLeft():
                actionTot = self.actor.getNextAction(nextStates=nextStates, epsilon=self.epsilon)  # a' <- P(s')
            self.actor.setETrace(state)  # e(s,a) <- 1
            TDError = self.critic.getTDError(self.board.getReinforcement(), state=state, stateTot=stateTot)  # tdError
            self.critic.setETrace(state)  # e(s) <- 1
            stateNumber = 0
            for s in currentEpisode:  # All (s,a) in current episode
                stateNumber += 1
                self.critic.updateExpValues(s, TDError, isCurrentState=s == state)  # V(s) <- V(s) + a TD e(s)
                self.critic.updateETrace(s)
                self.actor.updatePolicy(s, TDError)
                self.actor.updateETrace(s)
            state = deepcopy(stateTot)  # s <- s'
            action = deepcopy(actionTot)  # a <- a'
        currentEpisode.append(state)
        return currentEpisode

    def train(self):
        pinsLeft = []
        pinNr = []
        for i in range(self.numberOfEpisodes):
            self.runEpisode()
            pinsLeft.append(self.board.getPinsLeft())
            pinNr.append(i)
            self.epsilon *= self.epsilonDecay  # Decay epsilon
            if (i+1) % 25 == 0:
                print("Episode: " + str(i+1))
        print("End Epsilon: " + str(self.epsilon))
        font = {'size': 22}
        plt.rc("font", **font)
        plt.figure(figsize=(7, 3.5))
        plt.bar(pinNr, pinsLeft, width=1.0)
        plt.show()
        if self.showFinalSolution:
            self.epsilon = 0
            lastStateSet = self.runEpisode()
            animator = BoardAnimator(frameDelay=p["frameDelay"])
            animator.animateEpisode(boardSize=self.boardSize, episodeStates=lastStateSet,
                                    usingDiamondBoard=self.useDiamondBoard)
