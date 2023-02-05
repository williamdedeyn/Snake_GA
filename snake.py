import random
import numpy as np
import pygame
from settings import settings
from GA.neural_network import NeuralNetwork, NeuralNetworkViz


class Snake:

    def __init__(self,canvas):

        self.canvas = canvas
        self.nrows = settings['BOARD_SIZE'][0]
        self.ncols = settings['BOARD_SIZE'][1]
        self.width = settings['SNAKE_SIZE']
        self.color = settings['SNAKE_COLOR']
        self.vision = 8
        self.background = settings['SCREEN_COLOR']
        self.food_color = settings['FOOD_COLOR']
        self.directions_to_no = {
            (-1, 0): 0,
            (-1, -1): 1,
            (0, -1): 2,
            (1, -1): 3,
            (1, 0): 4,
            (1, 1): 5,
            (0, 1): 6,
            (-1, 1): 7,
        }
        self.no_to_direction = {
            0: (-1, 0),
            1: (-1, -1),
            2: (0, -1),
            3: (1, -1),
            4: (1, 0),
            5: (1, 1),
            6: (0, 1),
            7: (-1, 1),
        }
        self.direction = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        self.X = (random.randrange(3,self.ncols-3))*self.width
        self.Y = (random.randrange(3, self.nrows-3))*self.width

        self.brain = NeuralNetwork([32,20,12,4])
        self.snake = []
        self.total = 1
        self.iteration = 0
        self.hunger = settings['SNAKE_HUNGER']
        self.fitness = 0
        self.spawn_food()
        self.snake.append((self.X,self.Y))

    def reset_parameters(self):
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.X = (random.randrange(3, self.ncols - 3)) * self.width
        self.Y = (random.randrange(3, self.nrows - 3)) * self.width
        self.snake = []
        self.total = 1
        self.iteration = 0
        self.hunger = settings['SNAKE_HUNGER']
        self.spawn_food()
        self.snake.append((self.X, self.Y))

    def start_game(self):
        self.draw()

    def update(self,direction):
        self.iteration += 1
        self.hunger -= 1
        if self.check_direction(direction):
            self.direction = direction
            self.X = self.X + self.direction[0]*self.width
            self.Y = self.Y + self.direction[1]*self.width
            self.snake.append((self.X,self.Y))
            if len(self.snake) > self.total:
                del self.snake[0]

    def draw(self):
        self.canvas.fill(self.background)
        for coor in self.snake:
            pygame.draw.rect(self.canvas, self.color,pygame.Rect(coor[0],coor[1],self.width,self.width), 0)
        pygame.draw.rect(self.canvas, self.food_color, pygame.Rect(self.foodX, self.foodY, self.width, self.width), 0)

    def check_death(self,newX = None,newY = None):
        if newX is None and newY is None:
            newX = self.X
            newY = self.Y
        for coor in self.snake[:-1]:
            if coor == (newX,newY):
                return True
        if newX+ self.width > self.ncols*self.width or \
            newY + self.width > self.nrows*self.width or \
            newX < 0 or newY < 0 or self.hunger == 0:
            return True
        return False

    def eat_food(self):
        if self.X == self.foodX and self.Y == self.foodY:
            self.spawn_food()
            self.total += 1
            self.hunger = settings['SNAKE_HUNGER']
            return True
        return False

    def spawn_food(self):
        self.foodX = (random.randrange(0,self.ncols))*self.width
        self.foodY = (random.randrange(0, self.nrows))*self.width
        if (self.foodX,self.foodY) in self.snake:
            self.spawn_food()

    def check_direction(self,direction):
        if direction[0] + self.direction[0] == 0 and direction[0] != 0:
            return False
        if direction[1] + self.direction[1] == 0 and direction[1] != 0:
            return False
        return True

    def look_direction(self,direction):
        dx, dy = direction[0]*self.width, direction[1]*self.width
        x = self.X + dx
        y = self.Y + dy
        dist = 0
        dist_food = 0
        dist_tail = 0

        while ((0 <= x < self.ncols*self.width) and (0 <= y < self.nrows*self.width)):
            if x == self.foodX and y == self.foodY:
                dist_food = 1
            if (x,y) in self.snake[:-1]:
                dist_tail = 1
            x += dx
            y += dy
            dist += 1
        return np.array([dist, dist_food, dist_tail],dtype=float).reshape((3,1))

    def get_inputs(self):
        result = np.zeros((self.vision * 3 + 4+4,1),dtype= float)
        number = self.directions_to_no.get(self.direction)
        for i in range(4,12):
            num = (number + i)%8
            new_direction = self.no_to_direction.get(num)
            distances = self.look_direction(new_direction)
            result[i-4:i-4+3] = distances
        result[-8:-4] = self.get_direction_one_hot_encoded()
        result[-4:] = self.get_tail_direction()

        return result

    def get_direction_one_hot_encoded(self):
        return np.array([
            (0,-1) == self.direction,
            (0, 1) == self.direction,
            (-1, 0) == self.direction,
            (1, 0) == self.direction
            ],dtype=float).reshape((4,1))

    def get_tail_direction(self):
        if len(self.snake) == 1:
            return self.get_direction_one_hot_encoded()
        else:
            p2 = self.snake[-2]
            p1 = self.snake[-1]
            dx = p2[0]-p1[0]
            dy = p2[1] - p1[1]
            if dx < 0:
                return np.array([0,0,1,0],dtype=float).reshape((4,1))
            elif dx > 0:
                return np.array([0, 0, 0, 1], dtype=float).reshape((4, 1))
            elif dy < 0:
                return np.array([1, 0, 0, 0], dtype=float).reshape((4, 1))
            elif dy > 0:
                return np.array([0, 1, 0, 0], dtype=float).reshape((4, 1))

    def get_left(self):
        number = self.directions_to_no.get(self.direction)
        num = (number -2)%8
        direction = self.no_to_direction.get(num)
        return direction

    def get_right(self):
        number = self.directions_to_no.get(self.direction)
        num = (number + 2) % 8
        direction = self.no_to_direction.get(num)
        return direction

    def think(self,input):
        output = self.brain.feed_forward(input)
        index = np.argmax(output)
        direction = [(0,-1),(0,1),(-1,0),(1,0)][index]
        if self.check_direction(direction):
            return direction
        else:
            return self.direction

    def calculate_fitness(self):
        self.fitness = (self.iteration) + ((2**self.total) + (self.total**2.1)*500) - (((.25 * self.iteration)**1.3) * (self.total**1.2))