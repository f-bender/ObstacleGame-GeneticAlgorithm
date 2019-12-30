import pygame
from pipe import Pipe
from settings_constants import WIDTH, HEIGHT, BLACK, WHITE, thickness, POSX, POSY, defaultInterval, boxWidth, gap, thickness, defaultPipeSpeed, map, population_size
import os
import random
from box import Box
from mutation_crossover import get_new_population

pygame.init()

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (POSX,POSY)
gameDisplay = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Obstacle Dodge Game')
clock = pygame.time.Clock()
# font = pygame.font.Font(None, 60)

frameRate = 60
frameCount = 0
paintDelay = 1

pipeSpeed = defaultPipeSpeed
interval = defaultInterval
pipes = [Pipe(-thickness,thickness,random.randrange(WIDTH-gap),gap)]
boxes = [Box(x,boxWidth, boxWidth, [255*(1-x/population_size), 255*(x/population_size), 255*(x/population_size)]) for x in range(population_size)]
dead_boxes = []

exit = False
mouseX, mouseY = 0, 0
while not exit:
    frameCount += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
        if event.type == pygame.KEYDOWN:
            if event.unicode == "w":
                frameRate += 100
            if event.unicode == "s":
                frameRate = max(frameRate-100, 60)
            if event.unicode == "d":
                frameRate = 60
            if event.unicode == "q":
                paintDelay = max(paintDelay - 100, 1)
            if event.unicode == "e":
                paintDelay += 100
            print(frameRate)
            print(clock.get_fps())
        # print(event)

    gameDisplay.fill(BLACK)
    mouseX, mouseY = pygame.mouse.get_pos()

    if pipes[0].y > 0:
        pipes.insert(0, Pipe(pipes[0].y - interval, thickness, random.randrange(WIDTH-gap), gap))

    while True:
        if pipes[-1].y > HEIGHT:
            pipes.pop()
        else:
            break

    for pipe in pipes:
        pipe.move(pipeSpeed)
        pipe.draw(gameDisplay)

    for box in boxes:
        mouseX, mouseY = box.update(pipes, pipeSpeed)
        box.move()
        box.limit()
        for pipe in pipes:
            box.handleCollision(pipe)
            if not pipe.passed[box.id] and box.y+box.height < pipe.y:
                box.score += 1
                pipe.passed[box.id] = True
        if box.dead:
            boxes.remove(box)
            dead_boxes.append(box)
            continue
        colval = int(max( min(511, map( ((mouseX-box.x)**2+(mouseY-box.y)**2)**(0.5), 0, WIDTH*0.7, 511, 0)), 0))
        col = (min(511 - colval, 255), min(colval, 255), 0, 10)
        pygame.draw.line(gameDisplay, col, (mouseX, mouseY), (int(box.x+box.width//2), int(box.y+box.height//2)), box.width//5)
        box.paint(gameDisplay)
    
    if not boxes:
        # resets game
        # better = False
        # with open("best_brain\\stats.txt") as stats_file:
        #     stats_file.readline


        get_input = [(box.brain.weights_matrices,box.fitness) for box in dead_boxes]
        get_output = get_new_population( get_input )

        boxes = [Box(idx, boxWidth, boxWidth, [255*(1-idx/population_size), 255*(idx/population_size), 255*(idx/population_size)], weights_matrices=matrices) for idx, matrices in enumerate(get_output)]
        
        dead_boxes = []
        pipes = [Pipe(-thickness,thickness,random.randrange(WIDTH-gap),gap)]
        pipeSpeed = defaultPipeSpeed
        interval = defaultInterval
        continue

    # fps = font.render(str(int(clock.get_fps())), True, [255,0,0])
    # gameDisplay.blit(fps, (50, 50))

    pipeSpeed += defaultPipeSpeed / 4000
    if interval > 5*thickness+boxWidth:
        interval -= defaultInterval / 20000

    if frameCount%paintDelay == 0:
        pygame.display.update()
    clock.tick(frameRate)


pygame.quit()
quit()
