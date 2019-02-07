import tkinter
import time
import threading
import numpy as np
import simulation
from collections import deque

import csv
import datetime
import os
from shutil import copyfile

import read_joy as joycon
import pygame

class Event:
    DEBUG = False
    DEBUG = True

    USER_MODE = 'USER'
    RANDOM_MODE = 'RANDOM'

    IS_TRAJ = True

    DISP_SIZE = 1000
    CANVAS_MARGIN = 0
    # CANVAS_SIZE = DISP_SIZE - CANVAS_MARGIN*2

    CIRCLE_D = 30
    LINE_WIDTH = 2

    INTERACTIVE_TIME = 5*60
    TESTING_TIME = 15

    DIFF = 0.03
    h_speed = 0.05

    USER_COLOR = '#ff4500'
    COM_COLOR = '#32cd32'

    # 1:USER, 2:SYSTEM
    # INIT_POS1 = DISP_SIZE/2 - CIRCLE_D*2
    # INIT_POS2 = DISP_SIZE/2 + CIRCLE_D*2


    # Experimental numeric value...
    ARROW_KEYCODE = {'Enter':36, 'Up':111, 'Down':116, 'Right':114, 'Left':113}

    def __init__(self, mode=USER_MODE, testee=''):
        self.mode = mode
        self.system_mode = simulation.CNN_Simulator.IMITATION_MODE

        self.startup = False
        self.system_stop = False

        self.frame, self.canvas = [None]*2

        self.canvas_w, self.canvas_h = [None]*2
        self.init_pos1 = [None]*2
        self.init_pos2 = [None]*2

        self.is_output = False
        self.is_network_interval = False

        self.is_drawing = True

        # To responce for Key-Event at same-timing
        self.history = []

        self.TORUS = [False]*2

        self.testee = testee
        if self.testee == '':
            self.logfile = '../log.txt'
        else:
            self.logfile = '../log_' + self.testee + '.txt'

        pygame.init()
        self.joy = joycon.Joycon()

        animation_thread = threading.Thread(target=self.animation)
        animation_thread.start()

    def get_mode(self):
        return self.mode

    def get_init_pos(self):
        return (self.init_pos1, self.init_pos2)


    def animation(self):
        self.frame = tkinter.Tk()

        self.frame.title("Title")
        self.frame.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame.focus_set()
        self.frame.config(cursor='none')

        if not self.DEBUG or self.DEBUG:
            self.frame.attributes('-fullscreen', True)
            self.canvas_w = self.frame.winfo_screenwidth() - self.CANVAS_MARGIN*2
            self.canvas_h = self.frame.winfo_screenheight() - self.CANVAS_MARGIN*2
        else:
            self.canvas_w = self.frame.winfo_screenwidth()/2 - self.CANVAS_MARGIN*2
            self.canvas_h = self.frame.winfo_screenheight()/2 - self.CANVAS_MARGIN*2
            self.frame.geometry(str(int(self.canvas_w)+self.CANVAS_MARGIN*2)+'x'+str(int(self.canvas_h)+self.CANVAS_MARGIN*2))

        ################ DELETE ###################
        '''
        self.canvas_w = 500
        self.canvas_h = 500
        self.frame.geometry(str(500+self.CANVAS_MARGIN*2)+'x'+str(500+self.CANVAS_MARGIN*2))
        '''
        ###########################################

        print('w,h', self.canvas_w, self.canvas_h)

        # 1:USER, 2:SYSTEM
        self.init_pos1 = [self.canvas_w/2-self.CIRCLE_D/2-self.CIRCLE_D*3, self.canvas_h/2-self.CIRCLE_D/2]
        self.init_pos2 = [self.canvas_w/2-self.CIRCLE_D/2+self.CIRCLE_D*3, self.canvas_h/2-self.CIRCLE_D/2]

        # Initialize position
        self.x1_pos, self.y1_pos = self.init_pos1
        self.dx1, self.dy1 = self.DIFF, self.DIFF
        self.preget_pos1 = self.init_pos1

        self.x2_pos, self.y2_pos = self.init_pos2
        self.dx2, self.dy2 = [0]*2

        # self.canvas = tkinter.Canvas(self.frame, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, background='white')
        self.canvas = tkinter.Canvas(self.frame, width=self.canvas_w, height=self.canvas_h, background='black')
        # self.canvas.pack()
        self.canvas.place(x=self.CANVAS_MARGIN-1, y=self.CANVAS_MARGIN-1)

        self.frame.bind("<KeyPress>", self.keypress)
        self.frame.bind("<KeyRelease>", self.keyrelease)
        self.frame.bind("<FocusOut>", self.focusout)

        # circle1 = self.canvas.create_oval(self.init_pos1, self.init_pos1, self.init_pos1+self.CIRCLE_D, self.init_pos1+self.CIRCLE_D, fill=self.USER_COLOR, width=0)
        # circle2 = self.canvas.create_oval(self.init_pos2, self.init_pos2, self.init_pos2+self.CIRCLE_D, self.init_pos2+self.CIRCLE_D, fill=self.COM_COLOR, width=0)

        '''
        circle1 = self.canvas.create_oval(self.init_pos1[0], self.init_pos1[1], self.init_pos1[0]+self.CIRCLE_D, self.init_pos1[1]+self.CIRCLE_D, fill=self.USER_COLOR, width=0, tags='t_circle1')
        self.canvas.tag_bind('t_circle1')
        circle2 = self.canvas.create_oval(self.init_pos2[0], self.init_pos2[1], self.init_pos2[0]+self.CIRCLE_D, self.init_pos2[1]+self.CIRCLE_D, fill=self.COM_COLOR, width=0, tags='t_circle2')
        self.canvas.tag_bind('t_circle2')

        self.frame.update()
        while not self.ARROW_KEYCODE['Enter'] in self.history:
            self.frame.update()

            try:
                self.dx1, self.dy1 = np.array(self.joy.get_value())*5
            except pygame.error:
                print('Pygame ERROR')
                self.on_closing()
                pygame.quit()
                return

            self.x1_pos += self.dx1
            self.y1_pos += self.dy1
            self.canvas.move(circle1, self.dx1, self.dy1)
        '''
            
        '''
            # User Input with Arrow-Key
            for key in self.history:
                dx_, dy_ = self.dx1*100, self.dy1*100
                if key == self.ARROW_KEYCODE['Up']:
                    self.y1_pos -= self.dy1
                    if self.y1_pos < 0:
                        dy_ = self.y1_pos+self.dy1
                        self.y1_pos = 0
                    self.canvas.move(circle1, 0, -dy_)

                elif key == self.ARROW_KEYCODE['Down']:
                    self.y1_pos += self.dy1
                    if self.y1_pos > self.canvas_h-self.CIRCLE_D:
                        dy_ = self.canvas_h-self.CIRCLE_D - (self.y1_pos-self.dy1)
                        self.y1_pos = self.canvas_h-self.CIRCLE_D
                    self.canvas.move(circle1, 0, dy_)

                elif key == self.ARROW_KEYCODE['Left']:
                    self.x1_pos -= self.dx1
                    if self.x1_pos <= 0:
                        dx_ = self.x1_pos+self.dx1
                        self.x1_pos = 0
                    self.canvas.move(circle1, -dx_, 0)

                elif key == self.ARROW_KEYCODE['Right']:
                    self.x1_pos += self.dx1
                    if self.x1_pos >= self.canvas_w-self.CIRCLE_D:
                        dx_ = self.canvas_w-self.CIRCLE_D - (self.x1_pos-self.dx1)
                        self.x1_pos = self.canvas_w-self.CIRCLE_D
                    self.canvas.move(circle1, dx_, 0)

            self.dx1, self.dy1 = self.DIFF, self.DIFF
        '''

        '''
        self.canvas.delete('t_circle1')
        self.canvas.delete('t_circle2')
        self.init_position()
        circle1 = self.canvas.create_oval(self.init_pos1[0], self.init_pos1[1], self.init_pos1[0]+self.CIRCLE_D, self.init_pos1[1]+self.CIRCLE_D, fill=self.USER_COLOR, width=0, tags='t_circle1')
        self.canvas.tag_bind('t_circle1')
        circle2 = self.canvas.create_oval(self.init_pos2[0], self.init_pos2[1], self.init_pos2[0]+self.CIRCLE_D, self.init_pos2[1]+self.CIRCLE_D, fill=self.COM_COLOR, width=0, tags='t_circle2')
        self.canvas.tag_bind('t_circle2')

        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='3', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='2', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='1', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='START', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        '''

        # update_thread = threading.Thread(target=self.update, args=[(circle1, circle2)])

        update_thread = threading.Thread(target=self.update)
        # When this window is shutdowned, this thread is also finished.
        update_thread.daemon = True
        update_thread.start()

        # This thread must be worked in the background since it's infinite-loop.
        self.frame.mainloop()


    # Always Monitoring
    def update(self):
        if self.DEBUG:
            self.canvas.create_text(100, 50, justify='left', text='DEBUG', font=('FixedSys',24), tags='debug', fill='yellow')
            self.canvas.create_text(self.frame.winfo_screenwidth()-100, 50, justify='left', text='DEBUG', font=('FixedSys',24), tags='debug', fill='red')

            self.TESTING_TIME = 1

        # obj1, obj2 = obj
        prepos1, pre_pos2 = self.init_pos1, self.init_pos2
        R = self.CIRCLE_D/2

        # num of traj-line
        TRAJ_MAX = 30
        # traj-line length
        line_len = 20
        trajectory1, trajectory2 = deque([]), deque([])

        circle1 = self.canvas.create_oval(self.init_pos1[0], self.init_pos1[1], self.init_pos1[0]+self.CIRCLE_D, self.init_pos1[1]+self.CIRCLE_D, fill=self.USER_COLOR, width=0, tags='t_circle1')
        self.canvas.tag_bind('t_circle1')
        # circle2 = self.canvas.create_oval(self.init_pos2[0], self.init_pos2[1], self.init_pos2[0]+self.CIRCLE_D, self.init_pos2[1]+self.CIRCLE_D, fill=self.COM_COLOR, width=0, tags='t_circle2')
        # self.canvas.tag_bind('t_circle2')

        self.frame.update()

        print('Press Enter, Start')
        while not self.ARROW_KEYCODE['Enter'] in self.history:
            pass

        _t = 0
        start_time = datetime.datetime.now()
        while True:
            self.frame.update()

            # Time
            t = self.TESTING_TIME - int((datetime.datetime.now()-start_time).total_seconds())
            if t != _t:
                self.canvas.delete('time')
                self.canvas.create_text(self.canvas_w/2, 30, text='{minutes:02}:{seconds:02}'.format(minutes=int(t/60), seconds=t%60), font=('FixedSys',24), tags='time', fill='white')
                _t = t

            self.frame.update()

            try:
                self.dx1, self.dy1 = np.array(self.joy.get_value())*self.h_speed
            except pygame.error:
                print('Pygame ERROR')
                self.on_closing()
                pygame.quit()
                return

            if self.x1_pos+self.dx1 < 0:
                self.dx1 = -self.x1_pos
            elif self.x1_pos+self.dx1 > self.canvas_w-self.CIRCLE_D:
                self.dx1 = (self.canvas_w-self.CIRCLE_D)-self.x1_pos

            if self.y1_pos+self.dy1 < 0:
                self.dy1 = -self.y1_pos
            elif self.y1_pos+self.dy1 > self.canvas_h-self.CIRCLE_D:
                self.dy1 = (self.canvas_h-self.CIRCLE_D)-self.y1_pos

            self.x1_pos += self.dx1
            self.y1_pos += self.dy1
            self.canvas.move(circle1, self.dx1, self.dy1)

            if self.IS_TRAJ:
                diff = np.sqrt(sum((np.array(prepos1)-np.array([self.x1_pos, self.y1_pos]))**2))
                if diff >= line_len:
                    # _line = self.canvas.create_line(prepos1[0]+R, prepos1[1]+R, self.x1_pos+R, self.y1_pos+R, fill=self.USER_COLOR, width=self.LINE_WIDTH, dash=((3,3) if self.system_mode==simulation.CNN_Simulator.CREATIVE_MODE else ()))
                    _line = self.canvas.create_line(prepos1[0]+R, prepos1[1]+R, self.x1_pos+R, self.y1_pos+R, fill=self.USER_COLOR, width=self.LINE_WIDTH, dash=(3,3))
                    trajectory1.append(_line)
                    prepos1 = [self.x1_pos, self.y1_pos]
                    
                    if len(trajectory1) > TRAJ_MAX:
                        self.canvas.delete(trajectory1.popleft())

            t = int((datetime.datetime.now()-start_time).total_seconds()*1000)
            if t/1000 > self.TESTING_TIME:
                break

        # Delete & Initialize the trajectory
        while len(trajectory1) != 0:
            self.canvas.delete(trajectory1.popleft())
        prepos1 = self.init_pos1
        trajectory1 = deque([])

        self.canvas.delete('time')
        self.canvas.delete('t_circle1')
        # self.canvas.delete('t_circle2')
        self.init_position()
        circle1 = self.canvas.create_oval(self.init_pos1[0], self.init_pos1[1], self.init_pos1[0]+self.CIRCLE_D, self.init_pos1[1]+self.CIRCLE_D, fill=self.USER_COLOR, width=0, tags='t_circle1')
        self.canvas.tag_bind('t_circle1')
        circle2 = self.canvas.create_oval(self.init_pos2[0], self.init_pos2[1], self.init_pos2[0]+self.CIRCLE_D, self.init_pos2[1]+self.CIRCLE_D, fill=self.COM_COLOR, width=0, tags='t_circle2')
        self.canvas.tag_bind('t_circle2')

        print('Press Enter, Start')
        while not self.ARROW_KEYCODE['Enter'] in self.history:
            pass


        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='3', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')

        # Launch the system
        self.startup = True

        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='2', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='1', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.CANVAS_MARGIN+self.canvas_w/2, self.canvas_h/2, text='START', font=('FixedSys',36), tags='text', fill='white')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')


        f = open(self.logfile, mode='w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['testee1','W','H','CIRCLE_D','1:user,2:system','0:ch,1:im'])
        writer.writerow([self.testee,self.canvas_w,self.canvas_h,self.CIRCLE_D])
        writer.writerow(['time[ms]','mode','x1','y1','x2','y2','x1-x2','y1-y2'])
        start_time = datetime.datetime.now()

        
        _t = 0
        while True:
            self.frame.update()
            line_interval = self.DISP_SIZE/20

            # Time
            t = self.INTERACTIVE_TIME - int((datetime.datetime.now()-start_time).total_seconds())
            if t != _t:
                self.canvas.delete('time')
                if self.DEBUG:
                    self.canvas.create_text(self.canvas_w/2, 30, text='{minutes:02}:{seconds:02}:{mode:}'.format(minutes=int(t/60), seconds=t%60, mode='IMITATE' if self.system_mode else 'CHAOTIC'), font=('FixedSys',24), tags='time', fill='white')
                else:
                    self.canvas.create_text(self.canvas_w/2, 30, text='{minutes:02}:{seconds:02}'.format(minutes=int(t/60), seconds=t%60), font=('FixedSys',24), tags='time', fill='white')
                _t = t
                self.frame.update()
            
            # System-Output
            if self.is_output or self.is_network_interval:

                if self.is_network_interval:
                    self.is_network_interval = False
                    '''
                    print('network interval: ', self.wait_time/self.wait_time_count)
                    _wait_time_count += 1
                    time.sleep(self.wait_time/self.wait_time_count)
                    if self.wait_time_count == _wait_time_count:
                        self.is_network_interval = False
                        _wait_time_count = 0
                    '''
                if self.is_output:
                    self.is_output = False

                dx2_, dy2_ = self.dx2, self.dy2
                self.x2_pos += dx2_
                self.y2_pos += dy2_

                if self.x2_pos < 0:
                    # dx2_ = self.CANVAS_SIZE-self.CIRCLE_D - (self.x2_pos-dx2_)
                    # self.x2_pos = self.CANVAS_SIZE-self.CIRCLE_D
                    # dx2_ = self.x2_pos-self.dx2
                    # self.x2_pos = 0

                    # self.x2_pos -= dx2_*2
                    self.x2_pos -= dx2_
                    # pre_pos2[0] = self.x2_pos
                    dx2_ = -dx2_
                    dx2_ = 0
                elif self.x2_pos > self.canvas_w-self.CIRCLE_D:
                    # dx2_ = 0 - (self.x2_pos-dx2_)
                    # self.x2_pos = 0
                    # dx2_ = self.CANVAS_SIZE-self.CIRCLE_D - (self.x2_pos-dx2_)

                    # self.x2_pos = self.CANVAS_SIZE-self.CIRCLE_D
                    # self.x2_pos -= dx2_*2
                    self.x2_pos -= dx2_
                    # pre_pos2[0] = self.x2_pos
                    dx2_ = -dx2_
                    dx2_ = 0

                if self.y2_pos < 0:
                    # dy2_ = self.CANVAS_SIZE-self.CIRCLE_D - (self.y2_pos-dy2_)
                    # self.y2_pos = self.CANVAS_SIZE-self.CIRCLE_D

                    # self.y2_pos -= dy2_*2
                    self.y2_pos -= dy2_
                    # pre_pos2[1] = self.y2_pos
                    dy2_ = -dy2_
                    dy2_ = 0
                elif self.y2_pos > self.canvas_h-self.CIRCLE_D:
                    # dy2_ = 0 - (self.y2_pos-dy2_)
                    # self.y2_pos = 0
                    # self.y2_pos -= dy2_*2
                    self.y2_pos -= dy2_
                    # pre_pos2[1] = self.y2_pos
                    dy2_ = -dy2_
                    dy2_ = 0

                self.canvas.move(circle2, dx2_, dy2_)
                self.frame.update()

                # Drawing the Trajectory
                if self.IS_TRAJ:
                    diff = np.sqrt(sum((np.array(pre_pos2)-np.array([self.x2_pos, self.y2_pos]))**2))
                    if diff >= line_len:
                        # _line = self.canvas.create_line(pre_pos2[0]+R, pre_pos2[1]+R, self.x2_pos+R, self.y2_pos+R, fill=self.COM_COLOR, width=self.LINE_WIDTH, dash=((3,3) if self.system_mode==simulation.CNN_Simulator.CREATIVE_MODE else ()))
                        _line = self.canvas.create_line(pre_pos2[0]+R, pre_pos2[1]+R, self.x2_pos+R, self.y2_pos+R, fill=self.COM_COLOR, width=self.LINE_WIDTH, dash=(3,3))
                        trajectory2.append(_line)
                        pre_pos2 = [self.x2_pos, self.y2_pos]
                        
                        if len(trajectory2) > TRAJ_MAX:
                            self.canvas.delete(trajectory2.popleft())


            # Random Input
            if self.mode == self.RANDOM_MODE:
                line_interval = self.DISP_SIZE/40
                self.dx1 = (np.random.rand()-0.5)*10
                self.dy1 = (np.random.rand()-0.5)*10
                print(self.dx1, self.dy1)

                self.history.append(self.ARROW_KEYCODE['Right' if np.sign(self.dx1)>=1 else 'Left'])
                self.history.append(self.ARROW_KEYCODE['Down' if np.sign(self.dy1)>=1 else 'Up'])
                time.sleep(0.1)

            try:
                self.dx1, self.dy1 = np.array(self.joy.get_value())*self.h_speed
            except pygame.error:
                print('Pygame ERROR')
                self.on_closing()
                pygame.quit()
                return

            if self.x1_pos+self.dx1 < 0:
                self.dx1 = -self.x1_pos
            elif self.x1_pos+self.dx1 > self.canvas_w-self.CIRCLE_D:
                self.dx1 = (self.canvas_w-self.CIRCLE_D)-self.x1_pos

            if self.y1_pos+self.dy1 < 0:
                self.dy1 = -self.y1_pos
            elif self.y1_pos+self.dy1 > self.canvas_h-self.CIRCLE_D:
                self.dy1 = (self.canvas_h-self.CIRCLE_D)-self.y1_pos

            self.x1_pos += self.dx1
            self.y1_pos += self.dy1
            self.canvas.move(circle1, self.dx1, self.dy1)

            if self.IS_TRAJ:
                diff = np.sqrt(sum((np.array(prepos1)-np.array([self.x1_pos, self.y1_pos]))**2))
                if diff >= line_len:
                    # _line = self.canvas.create_line(prepos1[0]+R, prepos1[1]+R, self.x1_pos+R, self.y1_pos+R, fill=self.USER_COLOR, width=self.LINE_WIDTH, dash=((3,3) if self.system_mode==simulation.CNN_Simulator.CREATIVE_MODE else ()))
                    _line = self.canvas.create_line(prepos1[0]+R, prepos1[1]+R, self.x1_pos+R, self.y1_pos+R, fill=self.USER_COLOR, width=self.LINE_WIDTH, dash=(3,3))
                    trajectory1.append(_line)
                    prepos1 = [self.x1_pos, self.y1_pos]
                    
                    if len(trajectory1) > TRAJ_MAX:
                        self.canvas.delete(trajectory1.popleft())

            '''
            # User Input with Arrow-Key
            for key in self.history:
                dx_, dy_ = self.dx1, self.dy1
                # print('pos1: ', self.y1_pos)
                if key == self.ARROW_KEYCODE['Up']:
                    self.y1_pos -= self.dy1
                    if self.y1_pos < 0:
                        # dy_ = -(self.CANVAS_SIZE-self.CIRCLE_D - self.y1_pos)
                        # self.y1_pos = self.CANVAS_SIZE-self.CIRCLE_D
                        dy_ = self.y1_pos+self.dy1
                        self.y1_pos = 0
                        pre_pos1[1] = self.y1_pos
                        # self.TORUS[1] = True

                    self.canvas.move(circle1, 0, -dy_)

                elif key == self.ARROW_KEYCODE['Down']:
                    self.y1_pos += self.dy1
                    if self.y1_pos > self.canvas_h-self.CIRCLE_D:
                        # dy_ = 0 - self.y1_pos
                        # self.y1_pos = 0
                        dy_ = self.canvas_h-self.CIRCLE_D - (self.y1_pos-self.dy1)
                        self.y1_pos = self.canvas_h-self.CIRCLE_D
                        pre_pos1[1] = self.y1_pos
                        # self.TORUS[1] = True

                    self.canvas.move(circle1, 0, dy_)

                elif key == self.ARROW_KEYCODE['Left']:
                    self.x1_pos -= self.dx1
                    if self.x1_pos <= 0:
                        # dx_ = -(self.CANVAS_SIZE-self.CIRCLE_D - self.x1_pos)
                        # self.x1_pos = self.CANVAS_SIZE-self.CIRCLE_D
                        dx_ = self.x1_pos+self.dx1
                        self.x1_pos = 0
                        pre_pos1[0] = self.x1_pos
                        # self.TORUS[0] = True

                    self.canvas.move(circle1, -dx_, 0)

                elif key == self.ARROW_KEYCODE['Right']:
                    self.x1_pos += self.dx1
                    if self.x1_pos >= self.canvas_w-self.CIRCLE_D:
                        # dx_ = 0 - self.x1_pos
                        # self.x1_pos = 0
                        dx_ = self.canvas_w-self.CIRCLE_D - (self.x1_pos-self.dx1)
                        self.x1_pos = self.canvas_w-self.CIRCLE_D
                        pre_pos1[0] = self.x1_pos
                        # self.TORUS[0] = True

                    self.canvas.move(circle1, dx_, 0)

            self.dx1, self.dy1 = self.DIFF, self.DIFF
            '''

            t = int((datetime.datetime.now()-start_time).total_seconds()*1000)
            writer.writerow([t, int(self.system_mode), self.x1_pos, self.y1_pos, self.x2_pos, self.y2_pos, self.x1_pos-self.x2_pos, self.y1_pos-self.y2_pos])
            # self.canvas.delete('t_circle')
            if t/1000 > self.INTERACTIVE_TIME:
                break


            '''
            # Drawing the Trajectory
            if self.IS_TRAJ:
                diff = np.sqrt(sum((np.array(pre_pos1)-np.array([self.x1_pos, self.y1_pos]))**2))
                if diff >= line_interval:
                    trajectory1.append(self.canvas.create_line(pre_pos1[0]+R, pre_pos1[1]+R, self.x1_pos+R, self.y1_pos+R, fill='blue', width=self.LINE_WIDTH))
                    pre_pos1 = [self.x1_pos, self.y1_pos]
                    if len(trajectory1) > TRAJ_MAX:
                        self.canvas.delete(trajectory1.popleft())
            '''

        f.close()
        print('Output a logfile: ' + self.logfile)
        # Backup a logfile
        copyfile(self.logfile, self.logfile+'.bk')
        print('Copy the logfile: ' + self.logfile+'.bk')

        self.canvas.create_text(self.canvas_w/2, self.canvas_h/2, text='FINISH', font=('FixedSys',36), tags='text', fill='white')
        self.canvas.delete('time')
        self.system_stop = True
        print('Press Enter, Finish')
        while not self.ARROW_KEYCODE['Enter'] in self.history:
            time.sleep(0.01)
            pass
        self.on_closing()


    def get_startup_signal(self):
        return self.startup

    def get_systemstop_signal(self):
        return self.system_stop
    

    def set_system_mode(self, mode):
        self.system_mode = mode

    def get_diff(self):
        pos = [self.x1_pos, self.y1_pos]
        _pre_pos, _pos = np.array(self.preget_pos1), np.array(pos)

        # diff = np.where(self.TORUS,
        #         np.sign(-(_pos-_pre_pos))*(self.CANVAS_SIZE-abs(_pos-_pre_pos)),
        #         _pos-_pre_pos
        #         )
        diff = _pos-_pre_pos

        self.TORUS = [False, False]
        diff = [diff[0], -diff[1]]
        self.preget_pos1 = pos

        diff_pos = [self.x1_pos-self.x2_pos, self.y1_pos-self.y2_pos]

        return diff, diff_pos, self.is_drawing

    def init_position(self):
        self.x1_pos, self.y1_pos = self.init_pos1
        self.x2_pos, self.y2_pos = self.init_pos2


    def set_movement(self, pos, mag):
        dx_, dy_ = pos
        self.dx2, self.dy2 = dx_*mag, -dy_*mag
        self.is_output = True

    def set_network_interval(self):
        self.is_network_interval = True

    def keypress(self, event):
        # print(event.keycode)
        if not event.keycode in self.history:
            self.history.append(event.keycode)

    def keyrelease(self, event):
        # print(event.keycode)
        if event.keycode in self.history:
            self.history.pop(self.history.index(event.keycode))

    def focusout(self, event):
        self.history = []


    def on_closing(self):
        self.frame.quit()
        self.frame.destroy()

        self.is_drawing = False


if __name__ == '__main__':
    e = Event(Event.RANDOM_MODE)


