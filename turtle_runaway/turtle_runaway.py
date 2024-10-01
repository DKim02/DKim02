# -*- coding: utf-8 -*-
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
        self.end_image_label = None  # 엔딩 이미지 라벨 초기화

        # 이미지 크기 설정
        width, height = 250, 250  # 원하는 크기로 이미지 조정

        # 이미지 파일 경로 설정 (절대 경로)
        try:
            original_start_image = Image.open("C:/Users/kdg63/.spyder-py3/image/start_image.png")
            resized_start_image = original_start_image.resize((width, height))  # width와 height에 원하는 크기를 입력
            self.start_image = ImageTk.PhotoImage(resized_start_image)

            original_success_image = Image.open("C:/Users/kdg63/.spyder-py3/image/success_image.png")
            resized_success_image = original_success_image.resize((width, height))  # 원하는 크기 지정
            self.end_success_image = ImageTk.PhotoImage(resized_success_image)

            original_fail_image = Image.open("C:/Users/kdg63/.spyder-py3/image/fail_image.png")
            resized_fail_image = original_fail_image.resize((width, height))  # 원하는 크기 지정
            self.end_fail_image = ImageTk.PhotoImage(resized_fail_image)
            
            original_image = Image.open("C:/Users/kdg63/.spyder-py3/image/rabbit.gif")
            resized_image = original_image.resize((100, 100))  # 원하는 크기로 조정
            resized_image.save("C:/Users/kdg63/.spyder-py3/image/rabbit_resized.gif")  # 새로운 파일로 저장
        except Exception as e:
            print(f"Error loading images: {e}")
            self.start_image = None
            self.end_success_image = None
            self.end_fail_image = None

        # 시작 화면 이미지 표시
        if self.start_image:
            self.start_label = tk.Label(self.root, image=self.start_image)
            self.start_label.pack()

        # 시작 버튼 추가
        self.start_button = tk.Button(self.root, text="게임 시작", command=self.start_game)
        self.start_button.pack()

        # 조작하는 캐릭터 외형
        self.chaser.shape('turtle')
        self.chaser.color('red')
        self.chaser.penup()

        # 점수와 시간 표시용 거북이 추가
        self.drawer = turtle.RawTurtle(canvas)
        self.drawer.hideturtle()
        self.drawer.penup()

        self.timer_drawer = turtle.RawTurtle(canvas)
        self.timer_drawer.hideturtle()
        self.timer_drawer.penup()


    def start_game(self):
        # 시작 화면 요소 숨기기
        if self.start_image:
            self.start_label.pack_forget()
        self.start_button.pack_forget()
        self.game_started = True

        # 게임 시작
        self.start()

    def start(self, init_dist=400, ai_timer_msec=100):
        for i, runner in enumerate(self.runners):
            runner.setpos(random.randint(-init_dist // 2, init_dist // 2), random.randint(-init_dist // 2, init_dist // 2))
            runner.setheading(random.randint(0, 360))
        
        self.chaser.setpos((+init_dist / 2, 0))
        self.chaser.setheading(180)

        self.ai_timer_msec = ai_timer_msec
        self.time_left = self.time_limit  # 타이머 초기화
        self.canvas.ontimer(self.step, self.ai_timer_msec)

    def step(self):
        if not self.game_started:
            return

        # 각 도망자의 AI 동작 실행
        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i]:  
                runner.run_ai(self.chaser.pos(), self.chaser.heading())

        # 추격자 움직임 처리
        self.chaser.run_ai(None, None)

        # 각 도망자가 잡혔는지 확인
        for i, runner in enumerate(self.runners):
            if not self.caught_runners[i] and self.is_catched(runner):
                self.caught_runners[i] = True
                runner.hideturtle()  # 잡힌 도망자는 화면에서 사라짐

        # 남은 시간 갱신
        self.time_left -= self.ai_timer_msec / 1000

        # 화면에 점수(잡은 토끼 수)와 남은 시간 표시
        self.update_display()

        # 승리/패배 조건 체크
        if self.caught_runners.count(True) == 7:
            self.end_game(success=True)
        elif self.time_left <= 0:
            self.end_game(success=False)
        else:
            # 타이머가 계속 반복되도록 설정
            self.canvas.ontimer(self.step, self.ai_timer_msec)

    def update_display(self):
        # 남은 시간 표시
        self.timer_drawer.clear()
        self.timer_drawer.setpos(130, 300)
        self.timer_drawer.write(f'남은 시간: {int(self.time_left)}s // ', font=("Arial", 10, "bold"))
        
        # 잡은 토끼 수 표시
        self.drawer.clear()
        self.drawer.setpos(220, 300)
        self.drawer.write(f'잡은 토끼: {self.caught_runners.count(True)} / 7마리', font=("Arial", 10, "bold"))

    def is_catched(self, runner):
        p = runner.pos()
        q = self.chaser.pos()
        dx, dy = p[0] - q[0], p[1] - q[1]
        return dx**2 + dy**2 < self.catch_radius2

    def end_game(self, success):
        # 기존 요소들 정리
        self.drawer.clear()
        self.timer_drawer.clear()

        # 기존 end_image_label이 있으면 제거
        if self.end_image_label is not None:
            self.end_image_label.pack_forget()

        # 성공 또는 실패 화면 표시
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

        # 재시작 버튼 추가
        self.restart_button = tk.Button(self.root, text="재시작", command=self.restart_game)
        self.restart_button.pack()

    def restart_game(self):
        # 재시작 버튼 숨기기 및 엔딩 이미지 제거
        self.restart_button.pack_forget()
        if self.end_image_label:
            self.end_image_label.pack_forget()

        for runner in self.runners:
            runner.showturtle()  # 모든 도망자를 다시 보이게 함
        self.caught_runners = [False] * len(self.runners)
        self.start()

class ManualMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=10):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn
        self.penup()  # 자취를 남기지 않음

        # 키 입력 처리
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
        
        # 이미지 경로 설정
        image_path = "C:\\Users\\kdg63\\.spyder-py3\\image\\rabbit4.gif"

        # 이미지 등록
        screen.addshape(image_path)  # screen을 사용해 이미지 등록

        # 도망자 모양을 이미지로 변경
        self.shape(image_path)  # 등록된 이미지로 모양 설정
        
        # 도망자의 이동 및 회전 속도 설정
        self.step_move = random.randint(10, 20)  # 10에서 20 사이 무작위 속도
        self.step_turn = random.randint(10, 15)  # 10에서 15 사이 무작위 회전 속도
        self.penup()  # 자취를 남기지 않음




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
    root.title("Byeoljujeon")  # 여기서 창의 제목을 변경
    
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    screen = turtle.TurtleScreen(canvas)
    
    screen.bgpic("C:/Users/kdg63/.spyder-py3/image/background.gif")
    # 도망자 7명 생성
    runners = [RandomMover(screen) for _ in range(7)]
    chaser = ManualMover(screen)

    game = RunawayGame(root, screen, runners, chaser)
    screen.mainloop()