import tkinter
import time
import threading
import numpy as np
import simulation
from collections import deque


class Event():
    USER_MODE = 'USER'
    RANDOM_MODE = 'RANDOM'

    IS_TRAJ = False

    DISP_SIZE = 1000
    CANVAS_MARGIN = 10
    CANVAS_SIZE = DISP_SIZE - CANVAS_MARGIN*2

    CIRCLE_D = 30
    LINE_WIDTH = 2

    # 1:USER, 2:SYSTEM
    INIT_POS1 = DISP_SIZE/2 - CIRCLE_D*2
    INIT_POS2 = DISP_SIZE/2 + CIRCLE_D*2


    # Experimental numeric value...
    ARROW_KEYCODE = {'Up':111, 'Down':116, 'Right':114, 'Left':113}

    def __init__(self, mode=USER_MODE):
        self.mode = mode
        self.system_mode = simulation.CNN_Simulator.IMITATION_MODE

        self.frame, self.canvas = [None]*2

        self.x1_pos, self.y1_pos = [self.INIT_POS1]*2
        self.dx1, self.dy1 = 0.01, 0.01
        self.preget_pos1 = [self.INIT_POS1]*2

        self.x2_pos, self.y2_pos = [self.INIT_POS2]*2
        self.dx2, self.dy2 = [0]*2
        self.is_output = False

        self.is_drawing = True

        # To responce for Key-Event at same-timing
        self.history = []

        self.TORUS = [False]*2

        animation_thread = threading.Thread(target=self.animation)
        animation_thread.start()

    def get_mode(self):
        return self.mode


    def animation(self):
        self.frame = tkinter.Tk()
        self.frame.title("Title")
        self.frame.geometry(str(self.DISP_SIZE)+'x'+str(self.DISP_SIZE))
        self.frame.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame.focus_set()

        self.canvas = tkinter.Canvas(self.frame, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, background='white')
        # self.canvas.pack()
        self.canvas.place(x=self.CANVAS_MARGIN, y=self.CANVAS_MARGIN)

        self.frame.bind("<KeyPress>", self.keypress)
        self.frame.bind("<KeyRelease>", self.keyrelease)
        self.frame.bind("<FocusOut>", self.focusout)

        circle1 = self.canvas.create_oval(self.INIT_POS1, self.INIT_POS1, self.INIT_POS1+self.CIRCLE_D, self.INIT_POS1+self.CIRCLE_D, fill='blue', width=0)
        circle2 = self.canvas.create_oval(self.INIT_POS2, self.INIT_POS2, self.INIT_POS2+self.CIRCLE_D, self.INIT_POS2+self.CIRCLE_D, fill='red', width=0)

        update_thread = threading.Thread(target=self.update, args=[(circle1, circle2)])
        # When this window is shutdowned, this thread is also finished.
        update_thread.daemon = True
        update_thread.start()

        # This thread must be worked in the background since it's infinite-loop.
        self.frame.mainloop()

    # Always Monitoring
    def update(self, obj):
        obj1, obj2 = obj
        pre_pos1, pre_pos2 = [self.INIT_POS1]*2, [self.INIT_POS2]*2
        R = self.CIRCLE_D/2

        TRAJ_MAX = 50
        trajectory1, trajectory2 = deque([]), deque([])
        while True:
            line_interval = self.DISP_SIZE/20
            
            # System-Output
            if self.is_output:
                self.is_output = False

                dx2_, dy2_ = self.dx2, self.dy2
                self.x2_pos += dx2_
                self.y2_pos += dy2_

                if self.x2_pos < 0:
                    dx2_ = self.CANVAS_SIZE-self.CIRCLE_D - (self.x2_pos-dx2_)
                    self.x2_pos = self.CANVAS_SIZE-self.CIRCLE_D
                    pre_pos2[0] = self.x2_pos
                elif self.x2_pos > self.CANVAS_SIZE-self.CIRCLE_D:
                    dx2_ = 0 - (self.x2_pos-dx2_)
                    self.x2_pos = 0
                    pre_pos2[0] = self.x2_pos

                if self.y2_pos < 0:
                    dy2_ = self.CANVAS_SIZE-self.CIRCLE_D - (self.y2_pos-dy2_)
                    self.y2_pos = self.CANVAS_SIZE-self.CIRCLE_D
                    pre_pos2[1] = self.y2_pos
                elif self.y2_pos > self.CANVAS_SIZE-self.CIRCLE_D:
                    dy2_ = 0 - (self.y2_pos-dy2_)
                    self.y2_pos = 0
                    pre_pos2[1] = self.y2_pos

                self.canvas.move(obj2, dx2_, dy2_)

                # Drawing the Trajectory
                if self.IS_TRAJ:
                    diff = np.sqrt(sum((np.array(pre_pos2)-np.array([self.x2_pos, self.y2_pos]))**2))
                    if diff >= line_interval:
                        _line = self.canvas.create_line(pre_pos2[0]+R, pre_pos2[1]+R, self.x2_pos+R, self.y2_pos+R, fill='red', width=self.LINE_WIDTH, dash=((3,3) if self.system_mode==simulation.CNN_Simulator.CREATIVE_MODE else ()))
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


            # User Input with Arrow-Key
            for key in self.history:
                dx_, dy_ = self.dx1, self.dy1
                if key == self.ARROW_KEYCODE['Up']:
                    self.y1_pos -= self.dy1
                    if self.y1_pos <= 0:
                        dy_ = -(self.CANVAS_SIZE-self.CIRCLE_D - self.y1_pos)
                        self.y1_pos = self.CANVAS_SIZE-self.CIRCLE_D
                        pre_pos1[1] = self.y1_pos
                        self.TORUS[1] = True

                    self.canvas.move(obj1, 0, -dy_)

                elif key == self.ARROW_KEYCODE['Down']:
                    self.y1_pos += self.dy1
                    if self.y1_pos >= self.CANVAS_SIZE-self.CIRCLE_D:
                        dy_ = 0 - self.y1_pos
                        self.y1_pos = 0
                        pre_pos1[1] = self.y1_pos
                        self.TORUS[1] = True

                    self.canvas.move(obj1, 0, dy_)

                elif key == self.ARROW_KEYCODE['Left']:
                    self.x1_pos -= self.dx1
                    if self.x1_pos <= 0:
                        dx_ = -(self.CANVAS_SIZE-self.CIRCLE_D - self.x1_pos)
                        self.x1_pos = self.CANVAS_SIZE-self.CIRCLE_D
                        pre_pos1[0] = self.x1_pos
                        self.TORUS[0] = True

                    self.canvas.move(obj1, -dx_, 0)

                elif key == self.ARROW_KEYCODE['Right']:
                    self.x1_pos += self.dx1
                    if self.x1_pos >= self.CANVAS_SIZE-self.CIRCLE_D:
                        dx_ = 0 - self.x1_pos
                        self.x1_pos = 0
                        pre_pos1[0] = self.x1_pos
                        self.TORUS[0] = True

                    self.canvas.move(obj1, dx_, 0)

            self.dx1, self.dy1 = 0.01, 0.01

            # Drawing the Trajectory
            if self.IS_TRAJ:
                diff = np.sqrt(sum((np.array(pre_pos1)-np.array([self.x1_pos, self.y1_pos]))**2))
                if diff >= line_interval:
                    trajectory1.append(self.canvas.create_line(pre_pos1[0]+R, pre_pos1[1]+R, self.x1_pos+R, self.y1_pos+R, fill='blue', width=self.LINE_WIDTH))
                    pre_pos1 = [self.x1_pos, self.y1_pos]
                    if len(trajectory1) > TRAJ_MAX:
                        self.canvas.delete(trajectory1.popleft())

    def set_system_mode(self, mode):
        self.system_mode = mode

    def get_pos(self):
        pos = [self.x1_pos, self.y1_pos]
        _pre_pos, _pos = np.array(self.preget_pos1), np.array(pos)

        diff = np.where(self.TORUS,
                np.sign(-(_pos-_pre_pos))*(self.CANVAS_SIZE-abs(_pos-_pre_pos)),
                _pos-_pre_pos
                )

        self.TORUS = [False, False]
        diff = [diff[0], -diff[1]]
        self.preget_pos1 = pos

        return diff, self.is_drawing

    def set_movement(self, pos, mag):
        dx_, dy_ = pos
        self.dx2, self.dy2 = dx_*mag, -dy_*mag
        self.is_output = True

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


