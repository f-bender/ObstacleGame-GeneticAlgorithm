BLACK = (0,0,0)
WHITE = (255,255,255)

def map(val, min1,max1,min2,max2):
    return (val-min1)/(max1-min1)*(max2-min2)+min2

WIDTH = 800
HEIGHT = 600
POSY = 40
POSX = 2390

boxWidth = max(WIDTH // 48, HEIGHT // 27)
gap = max(WIDTH // 5, int(boxWidth*5))
thickness = HEIGHT // 20
defaultInterval = HEIGHT // 1.5
defaultPipeSpeed = HEIGHT // 360
damageFactor = 1600 / WIDTH
# txtSize = max(WIDTH / 40, HEIGHT / 22.5)


use_pretrained = True
save_directory = "exponential_fitness_overNextGap"
lower_y_higher_score = True

input_description = "[speedX_normalized, speedY_normalized, posX_normalized, pipeSpeed_normalized, gapX_normalized, dist_next_pipe_normalized, overnext_gapX_normalized]"

population_size = 100
assert population_size%2 == 0
mutation_rate = 0.5
variance = 0.2
default_hidden_layers = [7]
assert type(default_hidden_layers) is list
discourage_hitting_walls = True
fitness_calc = "exponential"
assert fitness_calc in ["exponential", "linear"]
activation = "sigmoid"
assert activation in ["sigmoid", "relu", "tanh"]

def calc_fitness(score, time_alive, life_left, wall_hit = False):
    if discourage_hitting_walls and wall_hit:
        return 0

    combined = score+time_alive/5000+life_left/50
    if fitness_calc == "linear":
        return combined
    if combined < 14.77:
        return combined
    return 1.2**(combined)