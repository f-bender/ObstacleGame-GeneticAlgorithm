import pygame
from pipe import Pipe
from settings_constants import WIDTH, HEIGHT, BLACK, WHITE, thickness, POSX, POSY, defaultInterval, boxWidth, gap, thickness, defaultPipeSpeed
from settings_constants import map, population_size, use_pretrained, save_directory, input_description
import os
import random
from box import Box
from mutation_crossover import get_new_population
import numpy

pygame.init()

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (POSX,POSY)
gameDisplay = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Obstacle Dodge Game')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 60)

frameRate = 60
frameCount = 0
paintDelay = 1
only_show_best = False

pipeSpeed = defaultPipeSpeed
interval = defaultInterval
pipes = [Pipe(-thickness,thickness,random.randrange(WIDTH-gap),gap)]
if use_pretrained:
    layers = 0
    with open(save_directory+"\\stats.txt", "r") as stats_file:
        lines = stats_file.readlines()
        for line in lines:
            line_elems = line.strip().split(": ")
            if line_elems[0] == "hidden_layers":
                layers = int(line_elems[-1])
        print(layers)

    brain_matrices = []
    for idx in range(layers+1):
        left = "input" if idx == 0 else f'hidden{idx}'
        right = "output" if idx == layers else f'hidden{idx+1}'
        brain_matrices.append(numpy.loadtxt(f'{save_directory}\\{left}-{right}.txt'))
    boxes = [Box(x,boxWidth, boxWidth, [255*(1-x/population_size), 255*(x/population_size), 255*(x/population_size)], weights_matrices=brain_matrices if x < population_size/2 else None) for x in range(population_size)]
else:
    boxes = [Box(x,boxWidth, boxWidth, [255*(1-x/population_size), 255*(x/population_size), 255*(x/population_size)]) for x in range(population_size)]
    # Create directory
    os.mkdir(save_directory) # error if directory exists (this is wanted behaviour)
    print("Directory " , save_directory ,  " created!") 

dead_boxes = []

exit = False
mouseX, mouseY = 0, 0
while not exit:
    # print(len(boxes))
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
            if event.unicode == "b":
                only_show_best = not only_show_best
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

    for idx, box in enumerate(boxes):
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
            if len(dead_boxes)==0 or dead_boxes[-1].fitness < box.fitness:
                dead_boxes.append(box)
            else:
                dead_boxes.insert(-1, box)

        if not only_show_best or idx == 0:
            colval = int(max( min(511, map( ((mouseX-box.x)**2+(mouseY-box.y)**2)**(0.5), 0, WIDTH*0.7, 511, 0)), 0))
            col = (min(511 - colval, 255), min(colval, 255), 0, 10)
            pygame.draw.line(gameDisplay, col, (mouseX, mouseY), (int(box.x+box.width//2), int(box.y+box.height//2)), box.width//5)
            box.paint(gameDisplay)

    
    if not boxes:
        # resets game
        better = False
        curr_fitness = 0
        try:
            with open(save_directory+"\\stats.txt", "r") as stats_file:
                curr_fitness = float(stats_file.readline().strip().split(": ")[-1])
        except:
            print(f'No stats in {save_directory} yet, creating them now!')
        
        if dead_boxes[-1].fitness > curr_fitness:
            for idx,matrix in enumerate(box.brain.weights_matrices):
                left = "input" if idx == 0 else f'hidden{idx}'
                right = "output" if idx == len(box.brain.weights_matrices)-1 else f'hidden{idx+1}'
                numpy.savetxt(f'{save_directory}\\{left}-{right}.txt', matrix)
            with open(save_directory+"\\stats.txt", "w") as stats_file:
                stats_file.write(f'fitness: {dead_boxes[-1].fitness}\n')
                stats_file.write(f'score: {dead_boxes[-1].score}\n')
                stats_file.write(f'time_alive: {dead_boxes[-1].time_alive}\n')
                stats_file.write(f'hidden_layers: {len(box.brain.weights_matrices)-1}\n')
                stats_file.write(f'input_description: {input_description}')
                
        


        get_input = [(box.brain.weights_matrices,box.fitness) for box in dead_boxes]
        get_output = get_new_population( get_input )

        boxes = [Box(idx, boxWidth, boxWidth, [255*(1-idx/population_size), 255*(idx/population_size), 255*(idx/population_size)], weights_matrices=matrices) for idx, matrices in enumerate(get_output)]
        
        dead_boxes = []
        pipes = [Pipe(-thickness,thickness,random.randrange(WIDTH-gap),gap)]
        pipeSpeed = defaultPipeSpeed
        interval = defaultInterval
        continue

    fps = font.render(str(boxes[0].score), True, [255,0,0])
    gameDisplay.blit(fps, (25, 50))

    pipeSpeed += defaultPipeSpeed / 4000
    if interval > 5*thickness+boxWidth:
        interval -= defaultInterval / 20000

    if frameCount%paintDelay == 0:
        pygame.display.update()
    clock.tick(frameRate)


pygame.quit()
quit()
