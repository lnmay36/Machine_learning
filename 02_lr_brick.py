'''
功能：使用pyGame实现一个简单的打砖块游戏
Created on Nov 30, 2012
@author: liury_lab

revised on Mar 2,2019
@reviser: Larry lyu
@description: 
    1、从cmd运行
    2、加入机器学习功能，让电脑自动玩游戏
'''
 
import pygame, sys, time, random    #@UnusedImport
from pygame.locals import *         #@UnusedWildImport
import matplotlib.pyplot as plt
import math
 
# 一些关于窗口的常量定义
WINDOW_WIDTH  = 640
WINDOW_HEIGHT = 480
SAFE_LINE = 390 #低于这条线就来不及预测了
PREDICT_LINE = 99 #高于这条线才进行预测
 
# 游戏状态常量定义
GAME_STATE_INIT        = 0
GAME_STATE_START_LEVEL = 1
GAME_STATE_RUN         = 2
GAME_STATE_GAMEOVER    = 3
GAME_STATE_SHUTDOWN    = 4
 
# 小球的常量定义
BALL_START_Y  = (WINDOW_HEIGHT//2)
BALL_SIZE     = 4
 
# 挡板的常量定义
PADDLE_START_X  = (WINDOW_WIDTH/2 - 16)
PADDLE_START_Y  = (WINDOW_HEIGHT - 32);
PADDLE_WIDTH    = 40
PADDLE_HEIGHT   = 8
 
# 砖块的常量定义
NUM_BLOCK_ROWS    = 6
NUM_BLOCK_COLUMNS = 8
BLOCK_WIDTH       = 64
BLOCK_HEIGHT      = 16
BLOCK_ORIGIN_X    = 8
BLOCK_ORIGIN_Y    = 8
BLOCK_X_GAP       = 80
BLOCK_Y_GAP       = 32
 
# 一些颜色常量定义
BACKGROUND_COLOR = (0, 0, 0)
BALL_COLOR       = (0, 0, 255)
PADDLE_COLOR     = (128, 64, 64)
BLOCK_COLOR      = (255, 128, 0)
TEXT_COLOR       = (255, 255, 255)
 
# 游戏的一些属性信息
TOTAL_LIFE       = 0
FPS              = 25
trainMinLoss     = 0.00001 #最小的可接受的损失
testMinLoss      = 0.00001
 
# 初始化砖块数组
def InitBlocks():
    #blocks = [[1] * NUM_BLOCK_COLUMNS] * NUM_BLOCK_ROWS
    blocks = []
    for i in range(NUM_BLOCK_ROWS):             #@UnusedVarialbe
        blocks.append([i+1] * NUM_BLOCK_COLUMNS)
    return blocks
 
# 检测小球是否与挡板或者砖块碰撞
def ProcessBall(blocks, ball_x, ball_y, paddle):
    if (ball_y > WINDOW_HEIGHT//2):
        if (ball_x+BALL_SIZE >= paddle['rect'].left and \
            ball_x-BALL_SIZE <= paddle['rect'].left+PADDLE_WIDTH and \
            ball_y+BALL_SIZE >= paddle['rect'].top and \
            ball_y-BALL_SIZE <= paddle['rect'].top+PADDLE_HEIGHT):
            return None
 
# 显示文字
def DrawText(text, font, surface, x, y):
    text_obj = font.render(text, 1, TEXT_COLOR)
    text_rect = text_obj.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_obj, text_rect)
 
# 退出游戏
def Terminate():
    pygame.quit()
    sys.exit()
    
# 等待用户输入
def WaitForPlayerToPressKey():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                Terminate()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    Terminate()
                return
 
# 游戏界面的初始化
pygame.init()
mainClock = pygame.time.Clock()
    
# 小球的位置和速度
ball_x  = 0
ball_y  = 0
ball_dx = 0
ball_dy = 0
 
# 挡板的运动控制
paddle_move_left  = False
paddle_move_right = False
 
# 挡板的位置和颜色
paddle  = {'rect' :pygame.Rect(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT), 
           'color': PADDLE_COLOR}
 
# 游戏状态
game_state  = GAME_STATE_INIT
blocks      = []
life_left   = TOTAL_LIFE
game_over   = False
blocks_hit  = 0
score       = 0
level       = 1
 
game_start_font = pygame.font.SysFont(None, 48)
game_over_font  = pygame.font.SysFont(None, 48)
text_font       = pygame.font.SysFont(None, 20)
 
#game_over_sound = pygame.mixer.Sound('gameover.wav')
#game_hit_sound = pygame.mixer.Sound('hit.wav')
#pygame.mixer.music.load('background.mp3')
 
windowSurface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
pygame.display.set_caption('打砖块')
 
 
DrawText('pyFreakOut', game_start_font, windowSurface, 
         (WINDOW_WIDTH/3), (WINDOW_HEIGHT/3 + 50))
DrawText('Press any key to start.', game_start_font, windowSurface, 
         (WINDOW_WIDTH/3)-60, (WINDOW_HEIGHT)/3+100)
pygame.display.update()
WaitForPlayerToPressKey()
 
 
# 播放背景音乐
#pygame.mixer.music.play(-1, 0.0)
 
# 机器学习部分，预测球落到安全线上时，对应的横坐标
predicted = False #是否预测
store_line = 1 #是否存储点的坐标
xVari = 0
yVari = 0
xmean = 0
ymean = 0
lr = 0.1
# 收集球所经过的坐标信息
points_y=[]
points_x=[]
normPoints_x = []
normPoints_y = []
trainData_x = []
trainData_y = []
testData_x = []
testData_y = []
def sample():
    global points_y,points_x
    points_y.append(ball_y)
    points_x.append(ball_x)
# 预测函数。样本特征是纵坐标，样本标签是横坐标（要预测的点）
def predict(feature, params):
    yhat = params[0] * feature + params[1]
    return yhat
# 损失函数
def computeLoss(train_y,train_x, params):#注意这里不能用points_x因为points_x/y可能被清空了，导致没数据出错
    loss = 0
    m = len(train_y)
    #print("数据点有%d" % m)
    for i in range(0,m):
        yhat = predict(train_y[i],params)
        loss += 0.5*(math.pow(yhat-train_x[i],2))   
        return loss/m

def computeGrad(train_y,train_x,params):#注意这里不能用points_x因为points_x/y可能被清空了，导致没数据除法出错
    wder = 0
    bder = 0
    m = len(train_y)
    for i in range(0,m):
        yhat = predict(train_y[i],params)
        wder += (yhat - train_x[i])*train_y[i]
        bder += (yhat - train_x[i])
    wder = wder/m;
    bder = bder/m;
    return wder,bder
def train(train_y,train_x,lr):  
    params = [1, 0]
    steps = 0
    while(computeLoss(train_y,train_x,params) > trainMinLoss and steps < 10000):
        wder,bder = computeGrad(train_y,train_x,params)
        params[0] = params[0] - lr*wder
        params[1] = params[1] - lr*bder
        steps += 1
    print("执行了 %d步，当前损失为:%f" % ( steps,computeLoss(train_y,train_x,params)))
    return params
def normalizeData():
    global xmean,ymean
    xmean = 0
    ymean = 0
    normPoints_x = []
    normPoints_y = []
    global trainData_x,trainData_y,testData_x,testData_y
    trainData_x = []
    trainData_y = []
    testData_x = []
    testData_y = []
    m = len(points_x)
    for i in range(0,m):
        xmean += points_x[i]
        ymean += points_y[i]
    xmean = xmean/m
    ymean = ymean/m
    global xVari,yVari
    xVari = 0
    yVari = 0
    for i in range(0,m):
        normPoints_x.append(points_x[i] - xmean)
        normPoints_y.append(points_y[i] - ymean)
    for i in range(0,m):
        xVari += math.pow(normPoints_x[i], 2)
        yVari += math.pow(normPoints_y[i], 2)
    xVari = math.sqrt(xVari)
    yVari = math.sqrt(yVari)
    up = int(m*0.7)
    if(xVari == 0):
        xVari =1
    if(yVari == 0):
        yVari =1
    for i in range(0,up):
        trainData_y.append(normPoints_y[i]/yVari)
        trainData_x.append(normPoints_x[i]/xVari)
    for i in range(up,m):
        testData_y.append(normPoints_y[i]/yVari)
        testData_x.append(normPoints_x[i]/xVari)

def test(params):
    testLoss = computeLoss(testData_y,testData_x, params)
    if(testLoss==0 or abs(testLoss - testMinLoss)<0.0001):
      return True
    else:
      return False
def trainAndPredict():
    global predicted,lr
    lr = 0.1
    if(len(points_x) >= 10 and ball_y>PREDICT_LINE):#在安全线以下才预测):
        normalizeData()
        params = train(trainData_y,trainData_x,lr)
        testPass= test(params)
        if(testPass):
            hitX = predict((PADDLE_START_Y-ymean)/yVari,params)*xVari + xmean
            predicted = True
            print("预测函数寻找成功，预计陨石降落点为: [%.2f]" %hitX)
            if(hitX > WINDOW_WIDTH or hitX < -1*BALL_SIZE):
                print("陨石不会降落到地球上，无需进行防卫")
            else:
                paddle['rect'].left=hitX - 14#- 24
                if paddle['rect'].left < 0:
                    paddle['rect'].left = 0
                elif paddle['rect'].left > WINDOW_WIDTH-PADDLE_WIDTH:
                    paddle['rect'].left = WINDOW_WIDTH-PADDLE_WIDTH
        else:
            #预测方法寻找失败，还需继续寻找
            lr = random.random()*0.1
            print("预测函数寻找失败，还需继续寻找")

# 游戏主循环
while True:
    # 事件监听
    for event in pygame.event.get():
        if event.type == QUIT:
            game_state = GAME_STATE_SHUTDOWN
            Terminate()
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                paddle_move_left = True
            if event.key == K_RIGHT:
                paddle_move_right = True
            if event.key == K_ESCAPE:
                game_state = GAME_STATE_SHUTDOWN
                Terminate()
        if event.type == KEYUP:
            if event.key == K_LEFT:
                paddle_move_left = False
            if event.key == K_RIGHT:
                paddle_move_right = False
   
    # 游戏控制流程           
    if game_state == GAME_STATE_INIT:
        # 初始化游戏
        predicted = False
        ball_x  = random.randint(8, WINDOW_WIDTH-8)
        ball_y  = BALL_START_Y
        points_y=[]
        points_x=[]
        if(store_line==1):
            sample()
        ball_dx = random.randint(-3, 4)
        ball_dy = random.randint( 5, 8)
        
        paddle['rect'].left = PADDLE_START_X
        paddle['rect'].top  = PADDLE_START_Y
        
        paddle_move_left  = False
        paddle_move_right = False
        
        life_left   = TOTAL_LIFE
        game_over   = False
        blocks_hit  = 0
        score       = 0
        level       = 1
        store_line = 1
        game_state  = GAME_STATE_START_LEVEL
    elif game_state == GAME_STATE_START_LEVEL:
        # 新的一关
        store_line = 1
        blocks = InitBlocks()
        game_state = GAME_STATE_RUN
    elif game_state == GAME_STATE_RUN:
        # 游戏运行
        
        # 球的运动
        ball_x += ball_dx;
        ball_y += ball_dy;
        
        #撞左右墙
        if ball_x > (WINDOW_WIDTH-BALL_SIZE) or ball_x < BALL_SIZE:
            print(ball_y)
            if ball_y > SAFE_LINE: #球太低不好预测，重新出球
                ball_x  = paddle['rect'].left + PADDLE_WIDTH // 2
                ball_y  = BALL_START_Y
                points_y=[]
                points_x=[]
                store_line = 1
                predicted = False
            else:
                ball_dx = -ball_dx
                ball_x  += ball_dx;
                points_y=[]
                points_x=[]
                store_line = 1
                predicted = False
        elif ball_y < BALL_SIZE:#天花板，pygame把左上角当成(0,0)
            ball_dy = -ball_dy
            ball_y  += ball_dy
            points_y=[]
            points_x=[]
            store_line = 1
            predicted = False
            
        elif ball_y > WINDOW_HEIGHT-BALL_SIZE: #失去防守，撞到底部底部
            if life_left == 0:
                game_state = GAME_STATE_GAMEOVER
            else:
                life_left -= 1
                # 初始化游戏
                ball_x  = paddle['rect'].left + PADDLE_WIDTH // 2
                ball_y  = BALL_START_Y
                ball_dx = random.randint(-4, 5)
                ball_dy = random.randint( 6, 9)
                points_y=[]
                points_x=[]
                store_line = 1
                predicted = False
            
        # 检测球是否与挡板碰撞
        if ball_y > WINDOW_HEIGHT // 2:
            if (ball_x+BALL_SIZE >= paddle['rect'].left and \
                ball_x-BALL_SIZE <= paddle['rect'].left+PADDLE_WIDTH and \
                ball_y+BALL_SIZE >= paddle['rect'].top and \
                ball_y-BALL_SIZE <= paddle['rect'].top+PADDLE_HEIGHT):
                ball_dy = - ball_dy
                ball_y += ball_dy
                #game_hit_sound.play()
                if paddle_move_left:
                    ball_dx -= random.randint(0, 3)
                elif paddle_move_right:
                    ball_dx += random.randint(0, 3)
                else:
                    ball_dx += random.randint(-1, 2)
                points_y=[]
                points_x=[]
                store_line = 0
                predicted = False
                    
        # 检测球是否与砖块碰撞
        cur_x = BLOCK_ORIGIN_X
        cur_y = BLOCK_ORIGIN_Y
        for row in range(NUM_BLOCK_ROWS):
            cur_x = BLOCK_ORIGIN_X
            for col in range(NUM_BLOCK_COLUMNS):
                if blocks[row][col] != 0:
                    if (ball_x+BALL_SIZE >= cur_x and \
                        ball_x-BALL_SIZE <= cur_x+BLOCK_WIDTH and \
                        ball_y+BALL_SIZE >= cur_y and \
                        ball_y-BALL_SIZE <= cur_y+BLOCK_HEIGHT):
                        blocks[row][col] = 0
                        blocks_hit += 1
                        ball_dy = -ball_dy
                        ball_dx += random.randint(-1, 2)
                        score += 5 * (level + abs(ball_dx))
                        #game_hit_sound.play()
                        points_y=[]
                        points_x=[]
                        store_line = 1
                        predicted = False
                cur_x += BLOCK_X_GAP
            cur_y += BLOCK_Y_GAP
            
        if blocks_hit == NUM_BLOCK_ROWS * NUM_BLOCK_COLUMNS:
            level       += 1
            blocks_hit  = 0
            score       += 1000
            game_state  = GAME_STATE_START_LEVEL
        
        # 人工智能玩游戏
        if(store_line==1):
            if (predicted==False and ball_y>PREDICT_LINE):#在安全线以下才预测:
                sample()
                trainAndPredict()    
        # 手工控制挡板的运动
        if paddle_move_left:
            paddle['rect'].left -= 8
            if paddle['rect'].left < 0:
                paddle['rect'].left = 0
        if paddle_move_right:
            paddle['rect'].left += 8
            if paddle['rect'].left > WINDOW_WIDTH-PADDLE_WIDTH:
                paddle['rect'].left = WINDOW_WIDTH-PADDLE_WIDTH
        
        # 绘制过程
        windowSurface.fill(BACKGROUND_COLOR)
        # 绘制挡板
        pygame.draw.rect(windowSurface, paddle['color'], paddle['rect'])
        # 绘制小球
        pygame.draw.circle(windowSurface, BALL_COLOR, (ball_x, ball_y), 
                           BALL_SIZE, 0)
        # 绘制砖块
        cur_x = BLOCK_ORIGIN_X
        cur_y = BLOCK_ORIGIN_Y
        for row in range(NUM_BLOCK_ROWS):
            cur_x = BLOCK_ORIGIN_X
            for col in range(NUM_BLOCK_COLUMNS):
                if blocks[row][col] != 0:
                    pygame.draw.rect(windowSurface, BLOCK_COLOR, 
                                     (cur_x, cur_y, BLOCK_WIDTH, BLOCK_HEIGHT))
                cur_x += BLOCK_X_GAP
            cur_y += BLOCK_Y_GAP
            
        # 绘制文字描述信息
        message = 'Level: ' + str(level) + '    Life: ' + str(life_left) + '    Score: ' + str(score)
        DrawText(message, text_font, windowSurface, 8, (WINDOW_HEIGHT - 16))
    elif game_state == GAME_STATE_GAMEOVER:
        DrawText('GAME OVER', game_over_font, windowSurface, 
                 (WINDOW_WIDTH / 3), (WINDOW_HEIGHT / 3))
        DrawText('Level: ' + str(level), game_over_font, windowSurface, 
                 (WINDOW_WIDTH / 3)+20, (WINDOW_HEIGHT / 3) + 50)
        DrawText('Score: ' + str(score), game_over_font, windowSurface, 
                 (WINDOW_WIDTH / 3)+20, (WINDOW_HEIGHT / 3) + 100)
        DrawText('Press any key to play again.', game_over_font, windowSurface, 
                 (WINDOW_WIDTH / 3)-80, (WINDOW_HEIGHT / 3) + 150)
        pygame.display.update()
        
        WaitForPlayerToPressKey()
        game_state = GAME_STATE_INIT

    pygame.display.update()
    mainClock.tick(FPS + level*2)



