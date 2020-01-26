import pygame
from pipe import Pipe
from settings_constants import WIDTH, HEIGHT, BLACK, thickness, POSX, POSY, defaultInterval, boxWidth, gap, defaultPipeSpeed, fitness_calc, discourage_hitting_walls
from settings_constants import mapValue, population_size, use_pretrained, save_directory, input_description, lower_y_higher_score, activation, default_hidden_layers, start_time
import os
import random
from box import Box
from mutation_crossover import get_new_population
import numpy
import datetime

def saveBox(box, directory):
    # save matrices defining the neural net
    for idx,matrix in enumerate(box.brain.weights_matrices):
        left = "input" if idx == 0 else f'hidden{idx}'
        right = "output" if idx == len(box.brain.weights_matrices)-1 else f'hidden{idx+1}'
        numpy.savetxt(f'{directory}\\{left}-{right}.txt', matrix)
    # save stats of the AI
    with open(directory+"\\stats.txt", "w") as stats_file:
        # stats of this particular AI on this particular run
        stats_file.write(f'fitness: {box.fitness}\n')
        stats_file.write(f'score for fitness calc: {box.score}\n')
        stats_file.write(f'score: {box.basic_score}\n')
        stats_file.write(f'time_alive: {box.time_alive}\n\n')
        # settings currently defined in settings_constants.py
        stats_file.write(f'hidden_layers: {len(box.brain.weights_matrices)-1}\n')
        stats_file.write(f'hidden_layer_layout: {str(default_hidden_layers)}\n')
        stats_file.write(f'input_description: {input_description}\n')
        stats_file.write(f'discourage_hitting_walls: {discourage_hitting_walls}\n')
        stats_file.write(f'activation: {activation}\n')
        stats_file.write(f'lower_y_higher_score: {lower_y_higher_score}\n')
        stats_file.write(f'fitness_calc: {fitness_calc}')


pygame.init()

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (POSX,POSY)
gameDisplay = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption(save_directory)
clock = pygame.time.Clock()
font = pygame.font.Font(None, 60)

frameRate = 60
frameCount = 0
paintDelay = 1
only_show_best = False
playing = False
player_box = None

pipeSpeed = defaultPipeSpeed*(1+start_time/4000)
interval = max(defaultInterval*(1-start_time/20000), 5*thickness+boxWidth)

pipes = [Pipe(-thickness,thickness,random.randrange(WIDTH-gap),gap)]
if use_pretrained:
    layers = 0
    with open(save_directory+"\\stats.txt", "r") as stats_file:
        lines = stats_file.readlines()
        for line in lines:
            line_elems = line.strip().split(": ")
            if line_elems[0] == "hidden_layers":
                layers = int(line_elems[-1])
            if line_elems[0] == "input_description":
                try:
                    assert line_elems[-1] == input_description
                except AssertionError as e:
                    print(str(e))
                    print(f'Inputs of saved: {line_elems[-1]}\nInputs currently: {input_description}')
                    exit(1)

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

paint = True
exitFlag = False
saving = False
mouseX, mouseY = 0, 0
while not exitFlag:
    frameCount += 1
    paint = frameCount%paintDelay == 0

    # handling input events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exitFlag = True
        if event.type == pygame.KEYDOWN:
            if event.unicode == "f":
                print("Current Framerate:",clock.get_fps())
            if event.unicode == "w":
                frameRate += 100
                print(f"Attempted framerate: {frameRate}")
            if event.unicode == "s":
                frameRate = max(frameRate-100, 60)
                print(f"Attempted framerate: {frameRate}")
            if event.unicode == "d":
                frameRate = 60
                paintDelay = 1
                only_show_best = False
                print(f"\nAttempted framerate: {frameRate}")
                print(f"Painting every {paintDelay}st frame")
                print("Painting","only best AI" if only_show_best else "every AI")
                print("You will"+("" if playing else " not")+" play next round\n")
            if event.unicode == "c":
                frameRate = 1560
                paintDelay = 1501
                only_show_best = True
                playing = False
                print(f"\nAttempted framerate: {frameRate}")
                print(f"Painting every {paintDelay}st frame")
                print("Painting","only best AI" if only_show_best else "every AI")
                print("You will"+("" if playing else " not")+" play next round\n")
            if event.unicode == "q":
                paintDelay = max(paintDelay - 100, 1)
                print(f"Painting every {paintDelay}st frame")
            if event.unicode == "e":
                paintDelay += 100
                print(f"Painting every {paintDelay}st frame")
            if event.unicode == "b":
                only_show_best = not only_show_best
                print("Painting","only best AI" if only_show_best else "every AI")
            if event.unicode == "k":
                saving = not saving
                print(("S" if saving else "Not s")+"aving best Brain after this Generation")
            if event.unicode == "p":
                playing = not playing
                print("You will"+("" if playing else " not")+" play next round")
                if not playing:
                    player_box = None

    # paint background
    gameDisplay.fill(BLACK)

    # create next pipe if the newest pipe is on screen
    if pipes[0].y > 0:
        pipes.insert(0, Pipe(pipes[0].y - interval, thickness, random.randrange(WIDTH-gap), gap))

    # remove all pipes that are offscreen (bottom)
    while True:
        if pipes[-1].y > HEIGHT:
            pipes.pop()
        else:
            break

    # handle pipe movement and drawing
    for pipe in pipes:
        pipe.move(pipeSpeed)
        if paint:
            pipe.draw(gameDisplay)

    # handle AI boxes: movement, drawing, collision
    for idx, box in enumerate(boxes):
        mouseX, mouseY = box.update(pipes, pipeSpeed)
        box.move()
        box.limit()
        for pipe in pipes:
            box.handleCollision(pipe)
            if not pipe.passed[box.id] and box.y+box.height < pipe.y:
                box.basic_score += 1
                if lower_y_higher_score:
                    box.score += 1-0.25*box.y/HEIGHT
                else:
                    box.score += 1
                pipe.passed[box.id] = True
        if box.dead:
            boxes.remove(box)
            if len(dead_boxes)==0 or dead_boxes[-1].fitness < box.fitness:
                dead_boxes.append(box)
            else:
                dead_boxes.insert(-1, box)
        if paint and (not only_show_best or idx == 0):
            box.paint(gameDisplay, mouseX, mouseY)

    # handle player box if playing
    if playing and player_box:
        mouseX, mouseY = player_box.update(pipes, pipeSpeed, playing=True)
        player_box.move()
        player_box.limit()
        for pipe in pipes:
            player_box.handleCollision(pipe)
            if not pipe.passed[player_box.id] and player_box.y+player_box.height < pipe.y:
                player_box.basic_score += 1
                pipe.passed[player_box.id] = True
        if player_box.dead:
            player_box = None
        elif paint:
            player_box.paint(gameDisplay, mouseX, mouseY)

    # if all boxes are dead
    if not boxes and not player_box:
        curr_fitness = 0
        # open the stats file of the current best AI
        try:
            with open(save_directory+"\\stats.txt", "r") as stats_file:
                curr_fitness = float(stats_file.readline().strip().split(": ")[-1])
        except:
            print(f'No stats in {save_directory} yet, creating them now!')
        
        # if the best fitness of the current population is better than the one currently saved in files, overwrite the files with the new best AI
        if dead_boxes[-1].fitness > curr_fitness:
            saveBox(dead_boxes[-1],save_directory)
            # # save matrices defining the neural net
            # for idx,matrix in enumerate(dead_boxes[-1].brain.weights_matrices):
            #     left = "input" if idx == 0 else f'hidden{idx}'
            #     right = "output" if idx == len(dead_boxes[-1].brain.weights_matrices)-1 else f'hidden{idx+1}'
            #     numpy.savetxt(f'{save_directory}\\{left}-{right}.txt', matrix)
            # # save stats of the AI
            # with open(save_directory+"\\stats.txt", "w") as stats_file:
            #     # stats of this particular AI on this particular run
            #     stats_file.write(f'fitness: {dead_boxes[-1].fitness}\n')
            #     stats_file.write(f'score for fitness calc: {dead_boxes[-1].score}\n')
            #     stats_file.write(f'score: {dead_boxes[-1].basic_score}\n')
            #     stats_file.write(f'time_alive: {dead_boxes[-1].time_alive}\n\n')
            #     # settings currently defined in settings_constants.py
            #     stats_file.write(f'hidden_layers: {len(dead_boxes[-1].brain.weights_matrices)-1}\n')
            #     stats_file.write(f'hidden_layer_layout: {str(default_hidden_layers)}\n')
            #     stats_file.write(f'input_description: {input_description}\n')
            #     stats_file.write(f'discourage_hitting_walls: {discourage_hitting_walls}\n')
            #     stats_file.write(f'activation: {activation}\n')
            #     stats_file.write(f'lower_y_higher_score: {lower_y_higher_score}\n')
            #     stats_file.write(f'fitness_calc: {fitness_calc}')
        elif saving:
            new_directory = save_directory+"\\"+str(datetime.datetime.now()).replace(":","-")
            os.mkdir(new_directory)
            saveBox(dead_boxes[-1],new_directory)
            saving = False
                
        # couple neural nets with their fitnesses
        get_input = [(box.brain.weights_matrices,box.fitness) for box in dead_boxes]
        # get new population
        get_output = get_new_population( get_input )
        # fill list of boxes with new population
        boxes = [Box(idx, boxWidth, boxWidth, [255*(1-idx/population_size), 255*(idx/population_size), 255*(idx/population_size)], weights_matrices=matrices) for idx, matrices in enumerate(get_output)]
        
        # reset: start a new game with the new population
        dead_boxes = []
        pipes = [Pipe(-thickness,thickness,random.randrange(WIDTH-gap),gap)]
        pipeSpeed = defaultPipeSpeed*(1+start_time/4000)
        interval = max(defaultInterval*(1-start_time/20000), 5*thickness+boxWidth)
        if playing:
            player_box = Box(-1, boxWidth, boxWidth, [255,0,255])
        continue

    if player_box:
        fps = font.render(str(player_box.basic_score), True, [255,0,0])
    else:
        fps = font.render(str(boxes[0].basic_score), True, [255,0,0])
    gameDisplay.blit(fps, (15, 15))

    # increase difficulty slightly on every frame
    pipeSpeed += defaultPipeSpeed / 2000 #4000
    if interval > 5*thickness+boxWidth:
        interval -= defaultInterval / 10000 #20000

    if paint:
        pygame.display.update()
    clock.tick(frameRate)


pygame.quit()
quit()

# def saveBox(box, directory):
#     # save matrices defining the neural net
#     for idx,matrix in enumerate(box.brain.weights_matrices):
#         left = "input" if idx == 0 else f'hidden{idx}'
#         right = "output" if idx == len(box.brain.weights_matrices)-1 else f'hidden{idx+1}'
#         numpy.savetxt(f'{directory}\\{left}-{right}.txt', matrix)
#     # save stats of the AI
#     with open(directory+"\\stats.txt", "w") as stats_file:
#         # stats of this particular AI on this particular run
#         stats_file.write(f'fitness: {box.fitness}\n')
#         stats_file.write(f'score for fitness calc: {box.score}\n')
#         stats_file.write(f'score: {box.basic_score}\n')
#         stats_file.write(f'time_alive: {box.time_alive}\n\n')
#         # settings currently defined in settings_constants.py
#         stats_file.write(f'hidden_layers: {len(box.brain.weights_matrices)-1}\n')
#         stats_file.write(f'hidden_layer_layout: {str(default_hidden_layers)}\n')
#         stats_file.write(f'input_description: {input_description}\n')
#         stats_file.write(f'discourage_hitting_walls: {discourage_hitting_walls}\n')
#         stats_file.write(f'activation: {activation}\n')
#         stats_file.write(f'lower_y_higher_score: {lower_y_higher_score}\n')
#         stats_file.write(f'fitness_calc: {fitness_calc}')


# if __name__ == "__main__":
#     main()