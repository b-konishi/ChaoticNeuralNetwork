import tkinter
import time
import threading


class Event():
    CIRCLE_D = 30
    CANVAS_PAD = 10
    DISP_SIZE = 400

    INIT_POS1 = 10
    INIT_POS2 = DISP_SIZE/2

    frame, canvas = None, None

    x1_pos, y1_pos = INIT_POS1, INIT_POS1
    dt = 0.01

    x2_pos, y2_pos = INIT_POS2, INIT_POS2
    dx2, dy2 = 0,0
    is_output = False

    history = []
    ARROW_KEYCODE = {'Up':111, 'Down':116, 'Right':114, 'Left':113}

    def __init__(self):
        animation_thread = threading.Thread(target=self.animation)
        animation_thread.start()


    def animation(self):
        self.frame = tkinter.Tk()
        self.frame.title("Title")
        self.frame.geometry(str(self.DISP_SIZE)+'x'+str(self.DISP_SIZE))
        self.frame.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.canvas = tkinter.Canvas(self.frame, width=self.DISP_SIZE, height=self.DISP_SIZE)
        # self.circle = self.canvas.create_oval(100,100,150,150, fill='blue', width=0)
        self.canvas.pack()

        self.frame.bind("<KeyPress>", self.keypress)
        self.frame.bind("<KeyRelease>", self.keyrelease)

        circle1 = self.canvas.create_oval(self.INIT_POS1, self.INIT_POS1, self.INIT_POS1+self.CIRCLE_D, self.INIT_POS1+self.CIRCLE_D, fill='red', width=0)
        circle2 = self.canvas.create_oval(self.INIT_POS2, self.INIT_POS2, self.INIT_POS2+self.CIRCLE_D, self.INIT_POS2+self.CIRCLE_D, fill='blue', width=0)

        update_thread = threading.Thread(target=self.update, args=[self.frame, (circle1, circle2)])
        update_thread.daemon = True
        update_thread.start()

        self.frame.mainloop()



    def update(self, frame, obj):
        obj1, obj2 = obj

        while True:
            if self.is_output:
                dx2_, dy2_ = 0,0

                dx2_, dy2_ = self.dx2, self.dy2
                print('update: ', dx2_, dy2_)
                self.x2_pos += dx2_
                self.y2_pos += dy2_

                '''
                if self.x2_pos <= 0 or self.x2_pos >= self.DISP_SIZE-self.CIRCLE_D:
                    self.x2_pos -= self.dx2
                    self.dx2 = 0.
                if self.y2_pos <= 0 or self.y2_pos >= self.DISP_SIZE-self.CIRCLE_D:
                    self.y2_pos -= self.dy2
                    self.dy2 = 0.
                '''

                if self.x2_pos <= 0:
                    dx2_ = self.DISP_SIZE-self.CIRCLE_D - self.x2_pos
                    self.x2_pos = self.DISP_SIZE-self.CIRCLE_D
                elif self.x2_pos >= self.DISP_SIZE-self.CIRCLE_D:
                    dx2_ = 0 - self.x2_pos
                    self.x2_pos = 0

                if self.y2_pos <= 0:
                    dy2_ = self.DISP_SIZE-self.CIRCLE_D - self.y2_pos
                    self.y2_pos = self.DISP_SIZE-self.CIRCLE_D
                elif self.y2_pos >= self.DISP_SIZE-self.CIRCLE_D:
                    dy2_ = 0 - self.y2_pos
                    self.y2_pos = 0

                print('POS: ', dx2_, dy2_)

                self.canvas.move(obj2, dx2_, dy2_)
                self.is_output = False

            dt_ = self.dt
            for key in self.history:
                if key == self.ARROW_KEYCODE['Up']:
                    self.y1_pos -= dt_
                    if self.y1_pos <= 0:
                        self.y1_pos += dt_
                        dt_ = 0
                    self.canvas.move(obj1, 0, -dt_)

                if key == self.ARROW_KEYCODE['Down']:
                    self.y1_pos += dt_
                    if self.y1_pos >= self.DISP_SIZE-self.CIRCLE_D:
                        self.y1_pos -= dt_
                        dt_ = 0
                    self.canvas.move(obj1, 0, dt_)

                if key == self.ARROW_KEYCODE['Left']:
                    self.x1_pos -= dt_
                    if self.x1_pos <= 0:
                        self.x1_pos += dt_
                        dt_ = 0
                    self.canvas.move(obj1, -dt_, 0)

                if key == self.ARROW_KEYCODE['Right']:
                    self.x1_pos += dt_
                    if self.x1_pos >= self.DISP_SIZE-self.CIRCLE_D:
                        self.x1_pos -= dt_
                        dt_ = 0
                    self.canvas.move(obj1, dt_, 0)

            # print('({},{})'.format(int(self.x1_pos), int(self.y1_pos)))
            # frame.update()

    def get_pos(self):
        return [self.x1_pos, self.y1_pos]

    def set_movement(self, pos):
        dx_, dy_ = pos
        self.dx2, self.dy2 = (dx_-0.5)*20, (dy_-0.5)*20
        self.is_output = True

    def keypress(self, event):
        # print(event.keycode)
        if not event.keycode in self.history:
            self.history.append(event.keycode)

    def keyrelease(self, event):
        # print(event.keycode)
        if event.keycode in self.history:
            self.history.pop(self.history.index(event.keycode))

    def on_closing(self):
        self.frame.quit()
        self.frame.destroy()


# root.bind("<Up>", upkey_event)

# e = Event()


