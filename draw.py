import tkinter
import time
import threading


class Event():
    x_pos, y_pos = 0,0
    frame = None
    canvas = None
    circle = None

    history = []
    ARROW_KEYCODE = {'Up':111, 'Down':116, 'Right':114, 'Left':113}

    dt = 0.01

    is_close = False
    
    def __init__(self):
        self.frame = tkinter.Tk()
        self.frame.title("Title")
        self.frame.geometry("400x300")
        self.frame.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.canvas = tkinter.Canvas(self.frame, width=400, height=300)
        # self.circle = self.canvas.create_oval(100,100,150,150, fill='blue', width=0)
        self.canvas.pack()

        self.frame.bind("<KeyPress>", self.keypress)
        self.frame.bind("<KeyRelease>", self.keyrelease)

        '''
        self.frame.bind("<Up>", self.upkey_event)
        self.frame.bind("<Down>", self.downkey_event)
        self.frame.bind("<Left>", self.leftkey_event)
        self.frame.bind("<Right>", self.rightkey_event)
        '''

        items = self.make_player(num=2)
        '''
        thread = threading.Thread(target=self.update)
        thread.start()
        '''

        self.frame.mainloop()

    def make_player(self, num):
        self.circle = self.canvas.create_oval(10,10,20,20, fill='red', width=0)

        thread = threading.Thread(target=self.update, name="thread%d"%num)
        thread.start()


    def update(self):
        while(not self.is_close):
            for key in self.history:
                if key == self.ARROW_KEYCODE['Up']:
                    self.canvas.move(self.circle, 0, -self.dt)
                    self.y_pos += self.dt
                if key == self.ARROW_KEYCODE['Down']:
                    self.canvas.move(self.circle, 0, self.dt)
                    self.y_pos -= self.dt
                if key == self.ARROW_KEYCODE['Left']:
                    self.canvas.move(self.circle, -self.dt, 0)
                    self.x_pos -= self.dt
                if key == self.ARROW_KEYCODE['Right']:
                    self.canvas.move(self.circle, self.dt, 0)
                    self.x_pos += self.dt

            print('({},{})'.format(int(self.x_pos), int(self.y_pos)))


    def keypress(self, event):
        print(event.keycode)
        if not event.keycode in self.history:
            self.history.append(event.keycode)

    def keyrelease(self, event):
        print(event.keycode)
        if event.keycode in self.history:
            self.history.pop(self.history.index(event.keycode))

    def on_closing(self):
        self.is_close = True
        self.frame.destroy()


# root.bind("<Up>", upkey_event)

e = Event()


