#!/usr/bin/python
# -*- Coding: utf-8 -*-

##
# ジョイスティック読み込み完成版
##

import pygame
from pygame.locals import *

class Joycon:
    FPS = 60

    axislist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rightaxis = [0.0, 0.0]

    def __init__(self):
        pygame.joystick.init()
        pygame.init()
        try:
            joys = pygame.joystick.Joystick(0)
            joys.init()
            # self.monitor()
            # print(self.get_value())
        except pygame.error:
            print('error has occured')
            print('It is possible to be disconnected...')
            pygame.quit()


    def event(self, e):
        if e.type == JOYAXISMOTION:
            #axes[e.axis] = e.value
            #print('{0:5d} axis {1:d}:{2: f}'.format(frame_no, e.axis, e.value))

            self.axislist[e.axis] = e.value
            print('leftstick | x | {0: .3f} | z | {1: .3f} || rightstick | x | {2: .3f} | z | {3: .3f}'.format(self.axislist[0], self.axislist[1], self.axislist[2], self.axislist[3]))

        elif e.type == JOYHATMOTION:
            pass
        elif e.type == JOYBUTTONUP:
            pass
        elif e.type == JOYBUTTONDOWN:
            pass

    def get_value(self):
        for e in pygame.event.get():
            if e.type == QUIT:
                return
            elif e.type == KEYDOWN and e.key == K_ESCAPE:
                return
            elif e.type == JOYAXISMOTION:
                self.axislist[e.axis] = e.value
                self.rightaxis = self.axislist[2:3+1]
                print('rightstick | x | {0:.3f} | z | {1:.3f}'.format(self.rightaxis[0], self.rightaxis[1]))

        return self.rightaxis

    def monitor(self):
        pygame.init()
        clock = pygame.time.Clock()
        clock.tick(self.FPS)
        frame_no = 0

        while True:
            for e in pygame.event.get():
                if e.type == QUIT:
                    return
                elif e.type == KEYDOWN and e.key == K_ESCAPE:
                    return
                else:
                    self.event(e)
            clock.tick(self.FPS)
            frame_no += 1

if __name__ == '__main__':
    joy = Joycon()





