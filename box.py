from pygame.draw import rect
from pygame.mouse import get_pos
from settings_constants import WHITE, map, damageFactor, WIDTH, HEIGHT, gap, thickness, defaultInterval, defaultPipeSpeed, default_hidden_layers, calc_fitness
from NeuralNetwork import NeuralNetwork
# import random

class Box:
  def __init__(self,id, w, h, color, weights_matrices = None):
    self.id = id
    self.width = w
    self.height = h
    self.x = WIDTH / 2 - self.width/2
    self.y = HEIGHT - self.height
    self.speedX = 0
    self.accX = 0
    self.speedY = 0
    self.accY = 0
    self.life = 100
    self.color = color
    self.dead = False
    self.score = 0
    self.time_alive = 0
    self.fitness = 0
    self.brain = NeuralNetwork(7,default_hidden_layers,2, weights_matrices = weights_matrices)


  def paint(self, surface):
    rect(surface, self.color, (self.x, self.y, self.width, self.height))

  def update(self, pipes, pipeSpeed):
    self.time_alive += 1
    speedX_normalized = map(self.speedX, -30,30,-1,1)
    speedY_normalized = map(self.speedY, -30,30,-1,1)
    posX_normalized = map(self.x+self.width/2, 0,WIDTH,-1,1)
    pipeSpeed_normalized = map(pipeSpeed, 0, 6*defaultPipeSpeed, -1,1)

    gapX_normalized = 0
    overnext_gapX_normalized = 0
    dist_next_pipe_normalized = 1

    pipes_ahead = [pipe for pipe in pipes[::-1] if pipe.y <= self.y]
    if len(pipes_ahead) > 0:
      gapX_normalized = map(pipes_ahead[0].gap_x, 0,WIDTH-gap,-1,1)
      dist_next_pipe_normalized = map(self.y-pipes_ahead[0].y-thickness, 0,defaultInterval-thickness,-1,1)
      if len(pipes_ahead) > 1:
        overnext_gapX_normalized = map(pipes_ahead[1].gap_x, 0,WIDTH-gap,-1,1)


    # dist_last_pipe_normalized = 1
    # try:
    #   last_pipe = next(pipe for pipe in pipes if pipe.y > self.y)
    #   dist_last_pipe_normalized = map(last_pipe.y-self.y-self.height, 0,defaultInterval-self.height,-1,1)
    # except:
    #   pass
      # print("no previous pipe"+str(random.randrange(10)))

    # TODO: add gapX of übernächste pipe! DONE!
    x, y = self.brain.predict_outputs([speedX_normalized,speedY_normalized,posX_normalized,pipeSpeed_normalized,
                                        gapX_normalized,dist_next_pipe_normalized,overnext_gapX_normalized])
    mouseX, mouseY = x*WIDTH/4+WIDTH/2, y*HEIGHT/4+HEIGHT/2
    # mouseX, mouseY = get_pos()
    self.accX = map(mouseX - (self.x+self.width/2), -1080, 1080, -2, 2)
    self.accY = map(mouseY - (self.y+self.height/2), -1080, 1080, -2, 2)
    return mouseX, mouseY

  def move(self):
    self.speedX += self.accX
    self.speedY += self.accY

    self.x += self.speedX
    self.speedX *= 0.99
    self.y += self.speedY
    self.speedY *= 0.99
    # print(self.speedX, self.speedY)
    

  def limit(self):
    if self.x < 0:
      self.x = 0
      self.life += self.speedX * damageFactor
      self.speedX = 0
    
    if self.x > WIDTH - self.width:
      self.x = WIDTH - self.width
      self.life -= self.speedX * damageFactor
      self.speedX = 0

    if self.y > HEIGHT - self.height:
      self.y = HEIGHT - self.height
      self.life -= self.speedY * damageFactor
      self.speedY = 0
    
    if self.y < 0:
      self.y = 0
      self.life += self.speedY * damageFactor
      self.speedY = 0

    if self.life <= 0:
      self.dead = True
      self.fitness = calc_fitness(self.score, self.time_alive)

  def handleCollision(self,pipe):
    # boxUpper = self.y
    # boxLower = self.y + self.height
    # boxLeft = self.x
    # boxRight = self.x + self.width
    # pipeUpper = pipe.y
    # pipeLower = pipe.y + pipe.thickness
    # pipeLeft = pipe.gap_x
    # pipeRight = pipe.gap_x + pipe.gap_width
    if self.y < pipe.y+pipe.thickness and self.y+self.height > pipe.y and (self.x < pipe.gap_x or self.x+self.width > pipe.gap_x+pipe.gap_width):
        self.dead = True
        self.fitness = calc_fitness(self.score, self.time_alive)