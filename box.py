from pygame.draw import rect, line
from pygame.mouse import get_pos
from settings_constants import mapValue, damageFactor, WIDTH, HEIGHT, WHITE, gap, thickness, defaultInterval, defaultPipeSpeed
from settings_constants import  default_hidden_layers, calc_fitness, input_description
from NeuralNetwork import NeuralNetwork
# import random

# TODO: Let every box have their own (self.)input_description!
# (so that boxes with different ones can compete)

class Box:
  def __init__(self,ID, w, h, color, weights_matrices = None):
    self.id = ID
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
    self.basic_score = 0
    self.time_alive = 0
    self.fitness = 0
    self.brain = NeuralNetwork(input_description.count(",")+1,default_hidden_layers,2, weights_matrices = weights_matrices)


  def paint(self, surface, mouseX = None, mouseY = None, xPos = None, yPos = None, border = False):
    if xPos is None:
      xPos = self.x
    if yPos is None:
      yPos = self.y
    if mouseX is not None and mouseY is not None:
      colval = int(max( min(511, mapValue( ((mouseX-xPos)**2+(mouseY-yPos)**2)**(0.5), 0, WIDTH*0.7, 511, 0)), 0))
      col = (min(511 - colval, 255), min(colval, 255), 0, 10)
      line(surface, col, (mouseX, mouseY), (int(xPos+self.width//2), int(yPos+self.height//2)), self.width//5)
    if border:
      rect(surface, WHITE, (xPos-1, yPos-1, self.width+2, self.height+2))
    rect(surface, self.color, (xPos, yPos, self.width, self.height))

  def update(self, pipes, pipeSpeed, playing = False):
    self.time_alive += 1
    mouseX, mouseY = 0,0
    if not playing:
      speedX_normalized = mapValue(self.speedX, -30,30,-1,1)
      speedY_normalized = mapValue(self.speedY, -30,30,-1,1)
      posX_normalized = mapValue(self.x+self.width/2, 0,WIDTH,-1,1)
      posY_normalized = mapValue(self.y+self.height/2, 0,HEIGHT,-1,1)
      pipeSpeed_normalized = mapValue(pipeSpeed, 0, 6*defaultPipeSpeed, -1,1)

      gapX_normalized = 0
      overnext_gapX_normalized = 0
      dist_next_pipe_normalized = 1

      pipes_ahead = [pipe for pipe in pipes[::-1] if pipe.y < self.y+self.height]
      if len(pipes_ahead) > 0:
        gapX_normalized = mapValue(pipes_ahead[0].gap_x, 0,WIDTH-gap,-1,1)
        dist_next_pipe_normalized = mapValue(self.y-pipes_ahead[0].y+self.height, 0,defaultInterval-thickness,-1,1)
        if len(pipes_ahead) > 1:
          overnext_gapX_normalized = mapValue(pipes_ahead[1].gap_x, 0,WIDTH-gap,-1,1)


      dist_last_pipe_normalized = 1
      try:
        last_pipe = next(pipe for pipe in pipes if pipe.y > self.y)
        dist_last_pipe_normalized = mapValue(last_pipe.y-self.y-self.height, 0,defaultInterval-self.height,-1,1)
      except:
        pass
        # print("no previous pipe"+str(random.randrange(10)))

      x, y = 0,0
      if input_description == "[speedX_normalized, speedY_normalized, posX_normalized, pipeSpeed_normalized, gapX_normalized, dist_next_pipe_normalized, overnext_gapX_normalized]":
        x, y = self.brain.predict_outputs([speedX_normalized,speedY_normalized,posX_normalized,pipeSpeed_normalized,
                                          gapX_normalized,dist_next_pipe_normalized,overnext_gapX_normalized])
      elif input_description == "[speedX_normalized, speedY_normalized, posX_normalized, pipeSpeed_normalized, gapX_normalized, dist_next_pipe_normalized, dist_last_pipe_normalized]":
        x, y = self.brain.predict_outputs([speedX_normalized,speedY_normalized,posX_normalized,pipeSpeed_normalized,
                                          gapX_normalized,dist_next_pipe_normalized,dist_last_pipe_normalized])
      elif input_description == "[speedX_normalized, speedY_normalized, posX_normalized, posY_normalized, pipeSpeed_normalized, gapX_normalized, dist_next_pipe_normalized, overnext_gapX_normalized]":
        x, y = self.brain.predict_outputs([speedX_normalized,speedY_normalized,posX_normalized,posY_normalized,pipeSpeed_normalized,
                                          gapX_normalized,dist_next_pipe_normalized,overnext_gapX_normalized])
      elif input_description == "[speedX_normalized, speedY_normalized, posX_normalized, posY_normalized, pipeSpeed_normalized, gapX_normalized, dist_next_pipe_normalized, dist_last_pipe_normalized, overnext_gapX_normalized]":
        x, y = self.brain.predict_outputs([speedX_normalized,speedY_normalized,posX_normalized,posY_normalized,pipeSpeed_normalized,
                                          gapX_normalized,dist_next_pipe_normalized, dist_last_pipe_normalized,overnext_gapX_normalized])                                 
      else:
        print(f'Unknown input description: {input_description}')
        exit(1)
      
      mouseX, mouseY = x*WIDTH/4+WIDTH/2, y*HEIGHT/4+HEIGHT/2
      # restrict mouse to be on screen (like a normal player)
      mouseX = min(max(mouseX, 0), WIDTH)
      mouseY = min(max(mouseY, 0), HEIGHT)
    else:
      mouseX, mouseY = get_pos()
    
    self.accX = mapValue(mouseX - (self.x+self.width/2), -1080, 1080, -2, 2)
    self.accY = mapValue(mouseY - (self.y+self.height/2), -1080, 1080, -2, 2)
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
      self.fitness = calc_fitness(self.score, self.time_alive, self.life, wall_hit = True)

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
        self.fitness = calc_fitness(self.score, self.time_alive, self.life)