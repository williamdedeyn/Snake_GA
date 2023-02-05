import pygame
import snake
from GA.population import Population
from settings import settings

N_ROWS, N_COLS = settings['BOARD_SIZE']
WIDTH_SNAKE = settings['SNAKE_SIZE']
POPULATION_SIZE = settings['POPULATION_SIZE']
SELECTION_AMOUNT = settings['PARENT_SIZE']
PROB_MUTATION = settings['MUTATE_PROB']
FRAMERATE = settings['SNAKE_SPEED']

pygame.init()
pygame.display.set_caption('Snake')
CANVAS = pygame.display.set_mode((N_COLS*WIDTH_SNAKE, N_ROWS*WIDTH_SNAKE))
clock = pygame.time.Clock()
running = True

snakes = [snake.Snake(CANVAS) for _ in range(POPULATION_SIZE)]
population = Population(snakes)

while running:
    print('Generation: ', population.generation)
    for snake in population.population:
        game = True
        snake.start_game()
        while game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    quit()

            new_direction = snake.think(snake.get_inputs())
            snake.update(new_direction)
            if snake.check_death():
                snake.calculate_fitness()
                snake.reset_parameters()
                game = False
            snake.eat_food()
            snake.draw()
            pygame.display.update()
            clock.tick(FRAMERATE)

    best_snakes = population.selection(SELECTION_AMOUNT)
    best_snake = best_snakes[0]
    new_population = population.reproduce(best_snakes,PROB_MUTATION)
    population.set_population(new_population)
    print("Best snake fitness: " ,best_snake.fitness)



