간단한 게임 규칙!
1. 간암에 시달리는 용왕의 부탁으로 거북이(자라아님)는 토끼의 간을 7개 모아서 소원을 이루어야 한다.
2. 정해지 시간 10초 이내에 토끼의 간을 7개를 수집하면 성공!
3. 만약, 정해진 시간안에 토끼를 잡지 못하면, 자라는 용봉탕이 된다...
4. 파이썬 코드는 아래와 같다.

Simple game rules!
1. At the request of the king of the dragon suffering from liver cancer, the turtle (not the turtle) must collect seven rabbits' livers and make a wish.
2. If you collect 7 rabbits' livers within 10 seconds of determination time, you're successful!
3. If you don't catch a rabbit within a set time, it becomes Yongbongtang...
4. The Python code is as below.


import tkinter as tk
import turtle, random
from PIL import Image, ImageTk

class RunawayGame:
    def __init__(self, root, canvas, runners, chaser, catch_radius=50):
        self.root = root
        self.canvas = canvas
        self.runners = runners
        self.chaser = chaser
        self.catch_radius2 = catch_radius**2
        self.caught_runners = [False] * len(runners)
        self.time_limit = 10
        self.time_left = self.time_limit
        self.game_started = False
        self.end_image_label = None  
        
        width, height = 250, 250
        
        try:
            original_start_image = Image.open("C:/Users/kdg63/.spyder-py3/image/start_image.png")
            resized_start_image = original_start_image.resize((width, height))  
            self.start_image = ImageTk.PhotoImage(resized_start_image)

            original_success_image = Image.open("C:/Users/kdg63/.spyder-py3/image/success_image.png")
            resized_success_image = original_success_image.resize((width, height))  
            self.end_success_image = ImageTk.PhotoImage(resized_success_image)

            original_fail_image = Image.open("C:/Users/kdg63/.spyder-py3/image/fail_image.png")
            resized_fail_image = original_fail_image.resize((width, height))  
            self.end_fail_image = ImageTk.PhotoImage(resized_fail_image)
            
            original_image = Image.open("C:/Users/kdg63/.spyder-py3/image/rabbit.gif")
            resized_image = original_image.resize((100, 100))  
            resized_image.save("C:/Users/kdg63/.spyder-py3/image/rabbit_resized.gif")  
        except Exception as e:
            print(f"Error loading images: {e}")
            self.start_image = None
            self.end_success_image = None
            self.end_fail_image = None

        if self.start_image:
            self.start_label = tk.Label(self.root, image=self.start_image)
            self.start_label.pack()

        self.start_button = tk.Button(self.root, text="게임 시작", command=self.start_game)
        self.start_button.pack()

        self.chaser.shape('turtle')
        self.chaser.color('red')
        self.chaser.penup()

        self.drawer = turtle.RawTurtle(canvas)
        self.drawer.hideturtle()
        self.drawer.penup()

        self.timer_drawer = turtle.RawTurtle(canvas)
        self.timer_drawer.hideturtle()
        self.timer_drawer.penup()


    def start_game(self):
        if self.start_image:
            self.start_label.pack_forget()
        self.start_button.pack_forget()
        self.game_started = True

        self.start()

    def start(self, init_dist=400, ai_timer_msec=100):
        for i, runner in enumerate(self.runners):
            runner.setpos(random.randint(-init_dist // 2, init_dist // 2), random.randint(-init_dist // 2, init_dist // 2))
            runner.setheading(random.randint(0, 360))
        
        self.chaser.setpos((+init_dist / 2, 0))
        self.chaser.setheading(180)

        self.ai_timer_msec = ai_timer_msec
        self.time_left = self.time_limit  
        self.canvas.ontimer(self.step, self.ai_timer_msec)

    def step(self):
        if not self.game_started:
            return

        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i]:  
                runner.run_ai(self.chaser.pos(), self.chaser.heading())

        self.chaser.run_ai(None, None)

        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i] and self.is_catched(runner):
                self.caught_runners[i] = True
                runner.hideturtle()  

        self.time_left -= self.ai_timer_msec / 1000

        self.update_display()

        if self.caught_runners.count(True) == 7:
            self.end_game(success=True)
        elif self.time_left <= 0:
            self.end_game(success=False)
        else:
            self.canvas.ontimer(self.step, self.ai_timer_msec)

    def update_display(self):
        self.timer_drawer.clear()
        self.timer_drawer.setpos(130, 300)
        self.timer_drawer.write(f'남은 시간: {int(self.time_left)}s // ', font=("Arial", 10, "bold"))
        
        self.drawer.clear()
        self.drawer.setpos(220, 300)
        self.drawer.write(f'잡은 토끼: {self.caught_runners.count(True)} / 7마리', font=("Arial", 10, "bold"))

    def is_catched(self, runner):
        p = runner.pos()
        q = self.chaser.pos()
        dx, dy = p[0] - q[0], p[1] - q[1]
        return dx**2 + dy**2 < self.catch_radius2

    def end_game(self, success):
        self.drawer.clear()
        self.timer_drawer.clear()

        if self.end_image_label is not None:
            self.end_image_label.pack_forget()

        if success:
            if self.end_success_image:
                self.end_image_label = tk.Label(self.root, image=self.end_success_image)
                self.end_image_label.pack()
            self.drawer.setpos(0, 0)
            self.drawer.write("You Win!", align="center", font=("Arial", 24, "bold"))
        else:
            if self.end_fail_image:
                self.end_image_label = tk.Label(self.root, image=self.end_fail_image)
                self.end_image_label.pack()
            self.drawer.setpos(0, 0)
            self.drawer.write("Time Over. You Dead!", align="center", font=("Arial", 24, "bold"))

        self.restart_button = tk.Button(self.root, text="재시작", command=self.restart_game)
        self.restart_button.pack()

    def restart_game(self):
        self.restart_button.pack_forget()
        if self.end_image_label:
            self.end_image_label.pack_forget()

        for runner in self.runners:
            runner.showturtle() 
        self.caught_runners = [False] * len(self.runners)
        self.start()

class ManualMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=10):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn
        self.penup()  

        canvas.onkeypress(lambda: self.forward(self.step_move), 'Up')
        canvas.onkeypress(lambda: self.backward(self.step_move), 'Down')
        canvas.onkeypress(lambda: self.left(self.step_turn), 'Left')
        canvas.onkeypress(lambda: self.right(self.step_turn), 'Right')
        canvas.listen()

    def run_ai(self, opp_pos, opp_heading):
        pass

class RandomMover(turtle.RawTurtle):
    def __init__(self, screen):
        super().__init__(screen)
        
        image_path = "C:\\Users\\kdg63\\.spyder-py3\\image\\rabbit4.gif"

        screen.addshape(image_path)  

        self.shape(image_path)  
        
        self.step_move = random.randint(10, 20) 
        self.step_turn = random.randint(10, 15)  
        self.penup() 




    def run_ai(self, opp_pos, opp_heading):
        mode = random.randint(0, 2)
        if mode == 0:
            self.forward(self.step_move)
        elif mode == 1:
            self.left(self.step_turn)
        elif mode == 2:
            self.right(self.step_turn)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Byeoljujeon")  
    
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    screen = turtle.TurtleScreen(canvas)
    
    screen.bgpic("C:/Users/kdg63/.spyder-py3/image/background.gif")
    
    runners = [RandomMover(screen) for _ in range(7)]
    chaser = ManualMover(screen)

    game = RunawayGame(root, screen, runners, chaser)
    screen.mainloop()