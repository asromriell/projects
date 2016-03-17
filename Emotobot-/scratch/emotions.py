__author__ = 'davidwen'

from enum import Enum


class Emotion(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    UNHAPPY = 4
    SURPRISE = 5
    NEUTRAL = 6


class Emotions:

    emotions = ['A', 'D', 'F', 'H', 'U', 'S', 'N']

    #def __init__(self):
        #self.Emotions = ['A', 'D', 'F', 'H', 'U', 'S', 'N']

    def build_vector(self, emotion):
        """
        build a one-hot vector indicating an emotion
        :param emotion: the (abbreviated) emotion to represent
        :return: a one-hot vector
        """
        try:
            idx = self.emotions.index(emotion.upper())
        except:
            print "%s not a defined emotion" % emotion
            return
        vector = [0 for i in range(7)]
        vector[idx] = 1
        return vector

if __name__ == '__main__':
    e = Emotions()
    print e.build_vector('a')
    print e.build_vector("D")
    print e.build_vector("f")
    print e.build_vector("h")
    print e.build_vector("N")
    print e.build_vector("u")
    print e.build_vector("s")
    print e.build_vector("sa")
