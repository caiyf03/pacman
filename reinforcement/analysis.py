# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    # answerNoise = 0.2
    # we make the pacman brave enough
    # we just make the noise small enough so that pacman will not afraid of dropping into -100 state
    answerNoise = 0
    return answerDiscount, answerNoise
# Prefer the close exit (+1), risking the cliff (-10)
# the discount should be large, the living should be very negative
# but should not too small that pacman just get down and die
def question3a():
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = -4
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Prefer the close exit (+1), but avoiding the cliff (-10)
def question3b():
    # discount should be very heavy
    # noise should be big but not too big
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Prefer the distant exit (+10), risking the cliff (-10)
def question3c():
    # to make pacman willing to go far, no discount is needed
    # a little living reward, pacman should not wonder
    # if living reward is 0 && discount is 1 after many iteration,
    # (4,1) will have 10, no matter east(stay still) or north (the exist with 10) the Q value is same
    # so since we use 'for' to find max it will choose the last one: east
    answerDiscount = 1
    answerNoise = 0
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Prefer the distant exit (+10), avoiding the cliff (-10)
def question3d():
    # we give it a little noise based on (c)
    answerDiscount = 1
    answerNoise = 0.5
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Avoid both exits and the cliff (so an episode should never terminate)
def question3e():
    # make pacman hesitating and no information
    answerDiscount = 0
    answerNoise = 1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():

    # with epsilon = 0 it will stick to what is best now and never go far away to explore new information

    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
