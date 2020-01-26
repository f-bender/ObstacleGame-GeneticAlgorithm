from settings_constants import WIDTH, WHITE, population_size
from pygame.draw import rect

class Pipe:
    
    def __init__(self, y, thickness, gap_x, gap_width):
        self.y = y
        self.thickness = thickness
        self.gap_x = gap_x
        self.gap_width = gap_width
        self.passed = [False for x in range(population_size+1)]

    def move(self, speed):
        self.y += speed

    def draw(self, surface):
        rect(surface, WHITE,(0,self.y, self.gap_x, self.thickness))
        rect(surface, WHITE,(self.gap_x+self.gap_width,self.y, WIDTH-self.gap_x-self.gap_width, self.thickness))
