<<<<<<< HEAD
간단한 게임 규칙!
1. 간암에 시달리는 용왕의 부탁으로 거북이(자라아님)는 토끼의 간을 7개 모아서 소원을 이루어야 한다.
2. 정해지 시간 10초 이내에 토끼의 간을 7개를 수집하면 성공!
3. 만약, 정해진 시간안에 토끼를 잡지 못하면, 자라는 용봉탕이 된다...
4. 방향키로 거북이를 움직이고 'T'를 눌러 텔레포트를 최대 3번 사용할 수 있다.
5. 주석을 포함한 파이썬 코드는 아래와 같다.

Simple game rules!
1. At the request of the king of the dragon suffering from liver cancer, the turtle (not the turtle) must collect seven rabbits' livers and make a wish.
2. If you collect 7 rabbits' livers within 10 seconds of determination time, you're successful!
3. If you don't catch a rabbit within a set time, it becomes Yongbongtang...
4. You can use the teleport up to 3 times by moving the turtle with the direction key and pressing 'T'.
5. The Python code with it's comments is as below.


import tkinter as tk
import turtle, random
from PIL import Image, ImageTk

# 게임 클래스
class RunawayGame:
    def __init__(self, root, canvas, runners, chaser, catch_radius=50):
        # 초기 설정
        self.root = root
        self.canvas = canvas
        self.runners = runners  # 도망가는 토끼들
        self.chaser = chaser    # 추적하는 거북이
        self.catch_radius2 = catch_radius**2  # 잡는 범위 (반경 제곱)
        self.caught_runners = [False] * len(runners)  # 잡힌 토끼들 여부
        self.time_limit = 10  # 제한 시간 설정
        self.time_left = self.time_limit  # 남은 시간
        self.teleport_count = 3  # 텔레포트 가능 횟수
        self.game_started = False  # 게임 시작 여부
        self.end_image_label = None  # 종료 이미지 라벨
        
        # 이미지 크기 설정
        width, height = 250, 250
        
        # 이미지 로딩 시도
        try:
            # 시작 이미지 로드
            original_start_image = Image.open("C:/Users/kdg63/.spyder-py3/image/start_image.png")
            resized_start_image = original_start_image.resize((width, height))  
            self.start_image = ImageTk.PhotoImage(resized_start_image)

            # 성공 이미지 로드
            original_success_image = Image.open("C:/Users/kdg63/.spyder-py3/image/success_image.png")
            resized_success_image = original_success_image.resize((width, height))  
            self.end_success_image = ImageTk.PhotoImage(resized_success_image)

            # 실패 이미지 로드
            original_fail_image = Image.open("C:/Users/kdg63/.spyder-py3/image/fail_image.png")
            resized_fail_image = original_fail_image.resize((width, height))  
            self.end_fail_image = ImageTk.PhotoImage(resized_fail_image)
            
            # 토끼 이미지 로드 및 크기 조정
            original_image = Image.open("C:/Users/kdg63/.spyder-py3/image/rabbit.gif")
            resized_image = original_image.resize((100, 100))  
            resized_image.save("C:/Users/kdg63/.spyder-py3/image/rabbit_resized.gif")  
        except Exception as e:
            print(f"Error loading images: {e}")
            self.start_image = None
            self.end_success_image = None
            self.end_fail_image = None

        # 시작 화면에 이미지 및 버튼 표시
        if self.start_image:
            self.start_label = tk.Label(self.root, image=self.start_image)
            self.start_label.pack()

        self.start_button = tk.Button(self.root, text="게임 시작", command=self.start_game)
        self.start_button.pack()

        # 추적자(거북이) 설정
        self.chaser.shape('turtle')
        self.chaser.color('red')
        self.chaser.penup()

        # 텍스트를 그리기 위한 거북이 설정
        self.drawer = turtle.RawTurtle(canvas)
        self.drawer.hideturtle()
        self.drawer.penup()

        # 타이머를 그리기 위한 거북이 설정
        self.timer_drawer = turtle.RawTurtle(canvas)
        self.timer_drawer.hideturtle()
        self.timer_drawer.penup()

        # 텔레포트 기능 키 't'에 연결
        canvas.onkeypress(self.teleport, 't')  
        canvas.listen()

    # 텔레포트 기능
    def teleport(self):
        if self.teleport_count > 0:  # 텔레포트 횟수가 남아 있을 때만 가능
            random_x = random.randint(-200, 200)  # 무작위 좌표
            random_y = random.randint(-200, 200)
            self.chaser.setpos(random_x, random_y)  # 거북이를 무작위 위치로 이동
            self.teleport_count -= 1  # 텔레포트 횟수 차감
            self.update_teleport_display()  # 텔레포트 횟수 업데이트

    # 게임 시작 버튼 클릭 시 호출
    def start_game(self):
        if self.start_image:
            self.start_label.pack_forget()
        self.start_button.pack_forget()
        self.game_started = True

        self.start()

    # 게임 로직 초기화
    def start(self, init_dist=400, ai_timer_msec=100):
        # 토끼들을 랜덤 위치로 배치
        for i, runner in enumerate(self.runners):
            runner.setpos(random.randint(-init_dist // 2, init_dist // 2), random.randint(-init_dist // 2, init_dist // 2))
            runner.setheading(random.randint(0, 360))
        
        # 거북이 초기 위치 설정
        self.chaser.setpos((+init_dist / 2, 0))
        self.chaser.setheading(180)

        # 텔레포트 횟수 초기화
        self.teleport_count = 3  
        self.update_teleport_display()  # 텔레포트 횟수 화면에 표시

        # 타이머 설정
        self.ai_timer_msec = ai_timer_msec
        self.time_left = self.time_limit  
        self.canvas.ontimer(self.step, self.ai_timer_msec)

    # 게임 상태 갱신
    def step(self):
        if not self.game_started:
            return

        # 토끼들 움직임 업데이트
        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i]:  
                runner.run_ai(self.chaser.pos(), self.chaser.heading())

        # 거북이 움직임 업데이트
        self.chaser.run_ai(None, None)

        # 토끼가 잡혔는지 확인
        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i] and self.is_catched(runner):
                self.caught_runners[i] = True
                runner.hideturtle()  # 잡힌 토끼 숨기기

        # 시간 감소
        self.time_left -= self.ai_timer_msec / 1000

        # 화면에 시간과 상태 업데이트
        self.update_display()

        # 게임 종료 조건 확인
        if self.caught_runners.count(True) == 7:
            self.end_game(success=True)
        elif self.time_left <= 0:
            self.end_game(success=False)
        else:
            self.canvas.ontimer(self.step, self.ai_timer_msec)

    # 게임 상태를 화면에 업데이트
    def update_display(self):
        # 시간과 잡은 토끼 수를 업데이트
        self.timer_drawer.clear()
        self.timer_drawer.setpos(130, 300)
        self.timer_drawer.write(f'남은 시간: {int(self.time_left)}s // ', font=("Arial", 10, "bold"))
    
        self.drawer.clear()
        self.drawer.setpos(220, 300)
        self.drawer.write(f'잡은 토끼: {self.caught_runners.count(True)} / 7마리', font=("Arial", 10, "bold"))
    
        # 텔레포트 횟수 업데이트도 호출
        self.update_teleport_display()

    # 텔레포트 횟수 화면에 업데이트
    def update_teleport_display(self):
        # 텔레포트 횟수를 좌측 상단에 표시
        self.drawer.setpos(-300, 300)  
        self.drawer.write(f'텔레포트 남은 횟수: {self.teleport_count}', font=("Arial", 10, "bold"))

    # 토끼가 잡혔는지 확인하는 메서드
    def is_catched(self, runner):
        p = runner.pos()
        q = self.chaser.pos()
        dx, dy = p[0] - q[0], p[1] - q[1]
        return dx**2 + dy**2 < self.catch_radius2

    # 게임 종료 처리
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

        # 재시작 버튼 표시
        self.restart_button = tk.Button(self.root, text="재시작", command=self.restart_game)
        self.restart_button.pack()

    # 재시작 처리
    def restart_game(self):
        self.restart_button.pack_forget()
        if self.end_image_label:
            self.end_image_label.pack_forget()

        for runner in self.runners:
            runner.showturtle() 
        self.caught_runners = [False] * len(self.runners)
        self.start()

# 수동으로 조작할 수 있는 거북이 클래스
class ManualMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=10):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn
        self.penup()  

        # 방향키로 이동
        canvas.onkeypress(lambda: self.forward(self.step_move), 'Up')
        canvas.onkeypress(lambda: self.backward(self.step_move), 'Down')
        canvas.onkeypress(lambda: self.left(self.step_turn), 'Left')
        canvas.onkeypress(lambda: self.right(self.step_turn), 'Right')
        canvas.listen()

    # AI 움직임 (현재는 없음)
    def run_ai(self, opp_pos, opp_heading):
        pass

# 랜덤하게 움직이는 토끼 클래스
class RandomMover(turtle.RawTurtle):
    def __init__(self, screen):
        super().__init__(screen)
        
        # 토끼 이미지 설정
        image_path = "C:\\Users\\kdg63\\.spyder-py3\\image\\rabbit4.gif"
        screen.addshape(image_path)  
        self.shape(image_path)  
        
        # 무작위 움직임 설정
        self.step_move = random.randint(10, 20) 
        self.step_turn = random.randint(10, 15)  
        self.penup() 

    # 토끼의 AI 움직임
    def run_ai(self, opp_pos, opp_heading):
        mode = random.randint(0, 2)
        if mode == 0:
            self.forward(self.step_move)
        elif mode == 1:
            self.left(self.step_turn)
        elif mode == 2:
            self.right(self.step_turn)

# 메인 코드
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Byeoljujeon")  
    
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    screen = turtle.TurtleScreen(canvas)
    
    # 배경 이미지 설정
    screen.bgpic("C:/Users/kdg63/.spyder-py3/image/background.gif")
    
    # 토끼 7마리 생성
    runners = [RandomMover(screen) for _ in range(7)]
    chaser = ManualMover(screen)

    # 게임 시작
    game = RunawayGame(root, screen, runners, chaser)
    screen.mainloop()
=======
간단한 게임 규칙!
1. 간암에 시달리는 용왕의 부탁으로 거북이(자라아님)는 토끼의 간을 7개 모아서 소원을 이루어야 한다.
2. 정해지 시간 10초 이내에 토끼의 간을 7개를 수집하면 성공!
3. 만약, 정해진 시간안에 토끼를 잡지 못하면, 자라는 용봉탕이 된다...
4. 방향키로 거북이를 움직이고 'T'를 눌러 텔레포트를 최대 3번 사용할 수 있다.
5. 주석을 포함한 파이썬 코드는 아래와 같다.

Simple game rules!
1. At the request of the king of the dragon suffering from liver cancer, the turtle (not the turtle) must collect seven rabbits' livers and make a wish.
2. If you collect 7 rabbits' livers within 10 seconds of determination time, you're successful!
3. If you don't catch a rabbit within a set time, it becomes Yongbongtang...
4. You can use the teleport up to 3 times by moving the turtle with the direction key and pressing 'T'.
5. The Python code with it's comments is as below.


import tkinter as tk
import turtle, random
from PIL import Image, ImageTk

# 게임 클래스
class RunawayGame:
    def __init__(self, root, canvas, runners, chaser, catch_radius=50):
        # 초기 설정
        self.root = root
        self.canvas = canvas
        self.runners = runners  # 도망가는 토끼들
        self.chaser = chaser    # 추적하는 거북이
        self.catch_radius2 = catch_radius**2  # 잡는 범위 (반경 제곱)
        self.caught_runners = [False] * len(runners)  # 잡힌 토끼들 여부
        self.time_limit = 10  # 제한 시간 설정
        self.time_left = self.time_limit  # 남은 시간
        self.teleport_count = 3  # 텔레포트 가능 횟수
        self.game_started = False  # 게임 시작 여부
        self.end_image_label = None  # 종료 이미지 라벨
        
        # 이미지 크기 설정
        width, height = 250, 250
        
        # 이미지 로딩 시도
        try:
            # 시작 이미지 로드
            original_start_image = Image.open("C:/Users/kdg63/.spyder-py3/image/start_image.png")
            resized_start_image = original_start_image.resize((width, height))  
            self.start_image = ImageTk.PhotoImage(resized_start_image)

            # 성공 이미지 로드
            original_success_image = Image.open("C:/Users/kdg63/.spyder-py3/image/success_image.png")
            resized_success_image = original_success_image.resize((width, height))  
            self.end_success_image = ImageTk.PhotoImage(resized_success_image)

            # 실패 이미지 로드
            original_fail_image = Image.open("C:/Users/kdg63/.spyder-py3/image/fail_image.png")
            resized_fail_image = original_fail_image.resize((width, height))  
            self.end_fail_image = ImageTk.PhotoImage(resized_fail_image)
            
            # 토끼 이미지 로드 및 크기 조정
            original_image = Image.open("C:/Users/kdg63/.spyder-py3/image/rabbit.gif")
            resized_image = original_image.resize((100, 100))  
            resized_image.save("C:/Users/kdg63/.spyder-py3/image/rabbit_resized.gif")  
        except Exception as e:
            print(f"Error loading images: {e}")
            self.start_image = None
            self.end_success_image = None
            self.end_fail_image = None

        # 시작 화면에 이미지 및 버튼 표시
        if self.start_image:
            self.start_label = tk.Label(self.root, image=self.start_image)
            self.start_label.pack()

        self.start_button = tk.Button(self.root, text="게임 시작", command=self.start_game)
        self.start_button.pack()

        # 추적자(거북이) 설정
        self.chaser.shape('turtle')
        self.chaser.color('red')
        self.chaser.penup()

        # 텍스트를 그리기 위한 거북이 설정
        self.drawer = turtle.RawTurtle(canvas)
        self.drawer.hideturtle()
        self.drawer.penup()

        # 타이머를 그리기 위한 거북이 설정
        self.timer_drawer = turtle.RawTurtle(canvas)
        self.timer_drawer.hideturtle()
        self.timer_drawer.penup()

        # 텔레포트 기능 키 't'에 연결
        canvas.onkeypress(self.teleport, 't')  
        canvas.listen()

    # 텔레포트 기능
    def teleport(self):
        if self.teleport_count > 0:  # 텔레포트 횟수가 남아 있을 때만 가능
            random_x = random.randint(-200, 200)  # 무작위 좌표
            random_y = random.randint(-200, 200)
            self.chaser.setpos(random_x, random_y)  # 거북이를 무작위 위치로 이동
            self.teleport_count -= 1  # 텔레포트 횟수 차감
            self.update_teleport_display()  # 텔레포트 횟수 업데이트

    # 게임 시작 버튼 클릭 시 호출
    def start_game(self):
        if self.start_image:
            self.start_label.pack_forget()
        self.start_button.pack_forget()
        self.game_started = True

        self.start()

    # 게임 로직 초기화
    def start(self, init_dist=400, ai_timer_msec=100):
        # 토끼들을 랜덤 위치로 배치
        for i, runner in enumerate(self.runners):
            runner.setpos(random.randint(-init_dist // 2, init_dist // 2), random.randint(-init_dist // 2, init_dist // 2))
            runner.setheading(random.randint(0, 360))
        
        # 거북이 초기 위치 설정
        self.chaser.setpos((+init_dist / 2, 0))
        self.chaser.setheading(180)

        # 텔레포트 횟수 초기화
        self.teleport_count = 3  
        self.update_teleport_display()  # 텔레포트 횟수 화면에 표시

        # 타이머 설정
        self.ai_timer_msec = ai_timer_msec
        self.time_left = self.time_limit  
        self.canvas.ontimer(self.step, self.ai_timer_msec)

    # 게임 상태 갱신
    def step(self):
        if not self.game_started:
            return

        # 토끼들 움직임 업데이트
        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i]:  
                runner.run_ai(self.chaser.pos(), self.chaser.heading())

        # 거북이 움직임 업데이트
        self.chaser.run_ai(None, None)

        # 토끼가 잡혔는지 확인
        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i] and self.is_catched(runner):
                self.caught_runners[i] = True
                runner.hideturtle()  # 잡힌 토끼 숨기기

        # 시간 감소
        self.time_left -= self.ai_timer_msec / 1000

        # 화면에 시간과 상태 업데이트
        self.update_display()

        # 게임 종료 조건 확인
        if self.caught_runners.count(True) == 7:
            self.end_game(success=True)
        elif self.time_left <= 0:
            self.end_game(success=False)
        else:
            self.canvas.ontimer(self.step, self.ai_timer_msec)

    # 게임 상태를 화면에 업데이트
    def update_display(self):
        # 시간과 잡은 토끼 수를 업데이트
        self.timer_drawer.clear()
        self.timer_drawer.setpos(130, 300)
        self.timer_drawer.write(f'남은 시간: {int(self.time_left)}s // ', font=("Arial", 10, "bold"))
    
        self.drawer.clear()
        self.drawer.setpos(220, 300)
        self.drawer.write(f'잡은 토끼: {self.caught_runners.count(True)} / 7마리', font=("Arial", 10, "bold"))
    
        # 텔레포트 횟수 업데이트도 호출
        self.update_teleport_display()

    # 텔레포트 횟수 화면에 업데이트
    def update_teleport_display(self):
        # 텔레포트 횟수를 좌측 상단에 표시
        self.drawer.setpos(-300, 300)  
        self.drawer.write(f'텔레포트 남은 횟수: {self.teleport_count}', font=("Arial", 10, "bold"))

    # 토끼가 잡혔는지 확인하는 메서드
    def is_catched(self, runner):
        p = runner.pos()
        q = self.chaser.pos()
        dx, dy = p[0] - q[0], p[1] - q[1]
        return dx**2 + dy**2 < self.catch_radius2

    # 게임 종료 처리
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

        # 재시작 버튼 표시
        self.restart_button = tk.Button(self.root, text="재시작", command=self.restart_game)
        self.restart_button.pack()

    # 재시작 처리
    def restart_game(self):
        self.restart_button.pack_forget()
        if self.end_image_label:
            self.end_image_label.pack_forget()

        for runner in self.runners:
            runner.showturtle() 
        self.caught_runners = [False] * len(self.runners)
        self.start()

# 수동으로 조작할 수 있는 거북이 클래스
class ManualMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=10):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn
        self.penup()  

        # 방향키로 이동
        canvas.onkeypress(lambda: self.forward(self.step_move), 'Up')
        canvas.onkeypress(lambda: self.backward(self.step_move), 'Down')
        canvas.onkeypress(lambda: self.left(self.step_turn), 'Left')
        canvas.onkeypress(lambda: self.right(self.step_turn), 'Right')
        canvas.listen()

    # AI 움직임 (현재는 없음)
    def run_ai(self, opp_pos, opp_heading):
        pass

# 랜덤하게 움직이는 토끼 클래스
class RandomMover(turtle.RawTurtle):
    def __init__(self, screen):
        super().__init__(screen)
        
        # 토끼 이미지 설정
        image_path = "C:\\Users\\kdg63\\.spyder-py3\\image\\rabbit4.gif"
        screen.addshape(image_path)  
        self.shape(image_path)  
        
        # 무작위 움직임 설정
        self.step_move = random.randint(10, 20) 
        self.step_turn = random.randint(10, 15)  
        self.penup() 

    # 토끼의 AI 움직임
    def run_ai(self, opp_pos, opp_heading):
        mode = random.randint(0, 2)
        if mode == 0:
            self.forward(self.step_move)
        elif mode == 1:
            self.left(self.step_turn)
        elif mode == 2:
            self.right(self.step_turn)

# 메인 코드
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Byeoljujeon")  
    
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    screen = turtle.TurtleScreen(canvas)
    
    # 배경 이미지 설정
    screen.bgpic("C:/Users/kdg63/.spyder-py3/image/background.gif")
    
    # 토끼 7마리 생성
    runners = [RandomMover(screen) for _ in range(7)]
    chaser = ManualMover(screen)

    # 게임 시작
    game = RunawayGame(root, screen, runners, chaser)
    screen.mainloop()
>>>>>>> 8f6f069e557ebe3ea76e10afc018cd4d60a64432
