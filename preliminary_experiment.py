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


class Event():
    CANVAS_MARGIN = 10
    # DISP_SIZE = 1000
    # CANVAS_SIZE = DISP_SIZE - CANVAS_MARGIN*2

    # interactive time[sec]
    INTERACTIVE_TIME = 5*60

    CIRCLE_D = 30
    DIFF = 10


    KEYCODE = {'Enter':36,
                'Up':111,'Down':116,'Right':114,'Left':113,
                'W':25,'S':39,'D':40,'A':38}

    def __init__(self):
        self.logfile = '../log.txt'

        self.testees = input('Input testees[testee1,testee2]: ').split(',')
        if len(self.testees) == 2:
            self.logfile = '../log_' + self.testees[0] + '_' + self.testees[1] + '.txt'
        else:
            self.testees = ['','']

        if os.path.isfile(self.logfile):
            ans = input('A logfile already exsts({}). Can I remove it?[y/n]: '.format(self.logfile))
            if not ans == 'y':
                print('Please BACKUP')
                exit(0)

        

        # To responce for Key-Event at same-timing
        self.history = []

        self.animation()

        # animation_thread = threading.Thread(target=self.animation)
        # animation_thread.start()


    def animation(self):
        # global frame, canvas, canvas_w, canvas_h, init_pos1, init_pos2, start_time

        self.frame = tkinter.Tk()
        self.frame.attributes('-fullscreen', True)
        self.frame.title("Preliminary experiment")
        # self.frame.geometry(str(self.DISP_SIZE)+'x'+str(self.DISP_SIZE))
        self.frame.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame.focus_set()
        self.frame.config(cursor='none')

        self.canvas_w = self.frame.winfo_screenwidth() - self.CANVAS_MARGIN*2
        self.canvas_h = self.frame.winfo_screenheight() - self.CANVAS_MARGIN*2

        print('w,h', self.canvas_w, self.canvas_h)

        self.init_pos1 = [self.canvas_w/2-self.CIRCLE_D*2, self.canvas_h/2-self.CIRCLE_D/2]
        self.init_pos2 = [self.canvas_w/2+self.CIRCLE_D*2, self.canvas_h/2-self.CIRCLE_D/2]

        self.canvas = tkinter.Canvas(self.frame, width=self.canvas_w, height=self.canvas_h, background='white')
        self.canvas.place(x=self.CANVAS_MARGIN, y=self.CANVAS_MARGIN)

        self.frame.bind("<KeyPress>", self.keypress)
        self.frame.bind("<KeyRelease>", self.keyrelease)
        self.frame.bind("<FocusOut>", self.focusout)

        self.frame.update()

        self.init_pos()
        while not self.KEYCODE['Enter'] in self.history:
            self.canvas.coords('t_circle', self.update())
            self.frame.update()
            time.sleep(0.020)

            self.canvas.delete('t_circle')


        self.canvas.create_text(self.canvas_w/2, self.canvas_h/2, text='3', font=('FixedSys',36), tags='text')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.canvas_w/2, self.canvas_h/2, text='2', font=('FixedSys',36), tags='text')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.canvas_w/2, self.canvas_h/2, text='1', font=('FixedSys',36), tags='text')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')
        self.canvas.create_text(self.canvas_w/2, self.canvas_h/2, text='START', font=('FixedSys',36), tags='text')
        self.frame.update()
        time.sleep(1)
        self.canvas.delete('text')


        self.init_pos()
        with open(self.logfile, mode='w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['testee1','testee2','W','H','CIRCLE_D','DIFF'])
            writer.writerow([self.testees[0],self.testees[1],self.canvas_w,self.canvas_h,self.CIRCLE_D,self.DIFF])
            writer.writerow(['time[ms]','x1','y1','x2','y2'])
            start_time = datetime.datetime.now()
            while True:
                self.canvas.delete('time')
                t = self.INTERACTIVE_TIME - int((datetime.datetime.now()-start_time).total_seconds())
                self.canvas.create_text(self.canvas_w/2, 30, text='{minutes:02}:{seconds:02}'.format(minutes=int(t/60), seconds=t%60), font=('FixedSys',24), tags='time')

                self.canvas.coords('t_circle', self.update())
                self.frame.update()
                time.sleep(0.020)

                t = int((datetime.datetime.now()-start_time).total_seconds()*1000)
                writer.writerow([t] + self.pos1 + self.pos2)
                self.canvas.delete('t_circle')
                if t/1000 > self.INTERACTIVE_TIME:
                    break
        print('Output a logfile: ' + self.logfile)

        # Backup a logfile
        copyfile(self.logfile, self.logfile+'.bk')
        print('Copy the logfile: ' + self.logfile+'.bk')
        

        self.canvas.create_text(self.canvas_w/2, self.canvas_h/2, text='FINISH', font=('FixedSys',36), tags='text')
        self.canvas.delete('time')
        while not self.KEYCODE['Enter'] in self.history:
            self.frame.update()

    # Always Monitoring
    def update(self):
        # canvas, circle1, circle2 = obj
        R = self.CIRCLE_D/2

        h = self.history.copy()
        for key in h:
            if key == self.KEYCODE['Up']:
                self.pos1[1] -= self.DIFF
            elif key == self.KEYCODE['Down']:
                self.pos1[1] += self.DIFF
            elif key == self.KEYCODE['Left']:
                self.pos1[0] -= self.DIFF
            elif key == self.KEYCODE['Right']:
                self.pos1[0] += self.DIFF

            elif key == self.KEYCODE['W']:
                self.pos2[1] -= self.DIFF
            elif key == self.KEYCODE['S']:
                self.pos2[1] += self.DIFF
            elif key == self.KEYCODE['A']:
                self.pos2[0] -= self.DIFF
            elif key == self.KEYCODE['D']:
                self.pos2[0] += self.DIFF

        # Edge processing
        self.pos1 = np.where(np.sign(self.pos1)==-1, 0, self.pos1)
        self.pos1 = np.where(np.array(self.pos1)>[self.canvas_w-self.CIRCLE_D,self.canvas_h-self.CIRCLE_D], [self.canvas_w-self.CIRCLE_D,self.canvas_h-self.CIRCLE_D], self.pos1)
        self.pos2 = np.where(np.sign(self.pos2)==-1, 0, self.pos2)
        self.pos2 = np.where(np.array(self.pos2)>[self.canvas_w-self.CIRCLE_D,self.canvas_h-self.CIRCLE_D], [self.canvas_w-self.CIRCLE_D,self.canvas_h-self.CIRCLE_D], self.pos2)

        self.pos1 = list(self.pos1)
        self.pos2 = list(self.pos2)

        self.canvas.create_oval(self.pos1[0], self.pos1[1], self.pos1[0]+self.CIRCLE_D, self.pos1[1]+self.CIRCLE_D, fill='red', width=0, tags='t_circle')
        self.canvas.create_oval(self.pos2[0], self.pos2[1], self.pos2[0]+self.CIRCLE_D, self.pos2[1]+self.CIRCLE_D, fill='blue', width=0, tags='t_circle')
        self.canvas.tag_bind('t_circle')

            
    def init_pos(self):
        self.pos1 = self.init_pos1
        self.pos2 = self.init_pos2


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


if __name__ == '__main__':
    e = Event()


