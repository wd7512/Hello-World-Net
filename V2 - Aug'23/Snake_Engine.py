import math
import random
import numpy as np

def index(b):
    return math.log(b)/math.log(2)

def bini(index):
    return 2 ** index

class Board():
    def __init__(self):
        self.size = 15
        self.full_size = self.size ** 2

        self.walls = 0

        for i in range(self.size):
            self.walls += bini(i)
            self.walls += bini(self.size*(self.size-1)+i)

        for i in range(self.size-2):
            self.walls += bini(self.size + self.size * i)
            self.walls += bini(self.size *2 -1 + self.size * i)

        self.food = 0
        self.head = bini(round(self.full_size / 2)) << self.size
        self.body_list = [(self.head >> self.size*2) , (self.head >> self.size)]

        self.food_points = 0
        self.move_points = 0
        self.energy = 100

        self.end = False

        self.update()

        self.place_food()

    def __str__(self):
        con = '{:'+str(self.full_size)+'b}'
        walls = con.format(self.walls)
        head = con.format(self.head)
        food = con.format(self.food)
        body = con.format(self.body)

        rows = []
        r = ''
        for i in range(self.full_size):
            r += ' '
            if walls[i] == '1':
                r = r + 'X'
            elif head[i] == '1':
                r = r + 'H'
            elif food[i] == '1':
                r = r + 'F'
            elif body[i] == '1':
                r = r + 'B'
            
            else:
                r = r + ' '

            if (i+1) % self.size == 0:
                rows.append(r[::-1])
                r = ''

        out = '\n'.join(rows)

        return out
        
    def update(self):
        self.body = sum(self.body_list)

        self.all = self.walls | self.food | self.body | self.head

        if self.head & self.body != 0:
            self.end = True
            #print('GAME OVER')

        if self.head & self.walls != 0:
            self.end = True
            #print('GAME OVER')

        if self.energy < self.move_points:
            self.end = True

    def place_food(self):
        choices = []
        for i in range(self.full_size):
            if bini(i) & self.all == 0:
                choices.append(i)

        loc = random.choice(choices)
        self.food = bini(loc)

    def print(self,var):
        con = '{:'+str(self.full_size)+'b}'
        return con.format(var)

    def push(self,move):
        # 0 - Left
        # 1 - UP
        # 2 - Right
        # 3 - DOWN

        old_head = self.head

        if move == 0:
            self.head = self.head >> 1
        elif move == 1:
            self.head = self.head << self.size
        elif move == 2:
            self.head = self.head << 1
        elif move == 3:
            self.head = self.head >> self.size
        else:
            self.end = True
            print('wtf is this input')

        if self.head & self.food == 0:#if not on a food
            self.body_list.remove(self.body_list[0])
            self.body_list.append(old_head)

        else:
            self.food_points += 1000
            self.energy += 100
            self.body_list.append(old_head)
            self.place_food()

        self.update()
        self.move_points += 1
    
    def get_inputs(self):
        out = np.array([[0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0]],dtype = float)

        if self.end == True:
            return out.flatten()
    
        # d / l / u / r
    
        head = index(self.head)
        food = index(self.food)
        
    
        head_loc = (int(head // self.size),int(head % self.size))
        food_loc = (int(food // self.size),int(food % self.size))
    
        diff_x = head_loc[0] - food_loc[0]
        diff_y = head_loc[1] - food_loc[1]
    
        #print(head_loc)
        #print(food_loc)
    
        ouot = out[0]
    
        ouot[0] = head_loc[0]
        ouot[1] = head_loc[1]
        ouot[2] = self.size - ouot[0] - 1
        ouot[3] = self.size - ouot[1] - 1
    
        ouot[4] = (ouot[0] + ouot[1] - 1) / 2
        ouot[5] = (ouot[1] + ouot[2] - 1) / 2
        ouot[6] = (ouot[2] + ouot[3] - 1) / 2
        ouot[7] = (ouot[3] + ouot[0] - 1) / 2
    
        if head_loc[0] == food_loc[0]:
    
            if head_loc[1] > food_loc[1]:
                ouot[8] = head_loc[1] - food_loc[1]
            else:
                ouot[9] = food_loc[1] - head_loc[1]
    
        if head_loc[1] == food_loc[1]:
            if head_loc[0] > food_loc[0]:
                ouot[10] = head_loc[0] - food_loc[0]
            else:
                ouot[11] = food_loc[0] - head_loc[0]
    
        
        if diff_x == diff_y:
            if diff_x < 0:
                ouot[12] = abs(diff_x)
            else:
                ouot[13] = abs(diff_x)
    
        if diff_x == -diff_y:
            if diff_x < 0:
                ouot[14] = abs(diff_x)
            else:
                ouot[15] = abs(diff_x)
    
    
        for body in self.body_list:
            b = index(body)
            loc = (int(b // self.size),int(b % self.size))
    
            diff_x = head_loc[0] - loc[0]
            diff_y = head_loc[1] - loc[1]
    
            if loc[0] == head_loc[0]:
                if loc[1] > head_loc[1]:
                    if ouot[16] == 0:
                        ouot[16] = loc[1] - head_loc[1]
                    else:
                        ouot[16] = min(ouot[16],loc[1] - head_loc[1])
    
                else:
                    if ouot[17] == 0:
                        ouot[17] = head_loc[1] - loc[1]
                    else:
                        ouot[17] = min(ouot[17],head_loc[1] - loc[1])
    
            if loc[1] == head_loc[1]:
                if loc[0] > head_loc[0]:
                    if ouot[18] == 0:
                        ouot[18] = loc[0] - head_loc[0]
                    else:
                        ouot[18] = min(ouot[18],loc[0] - head_loc[0])
    
                else:
                    if ouot[19] == 0:
                        ouot[19] = head_loc[0] - loc[0]
                    else:
                        ouot[19] = min(ouot[19],head_loc[0] - loc[0])
    
            if diff_x == diff_y:
                if diff_x < 0:
                    if ouot[20] == 0:
                        ouot[20] = abs(diff_x)
                    else:
                        ouot[20] = min(ouot[20],abs(diff_x))
                else:
                    if ouot[21] == 0:
                        ouot[21] = abs(diff_x)
                    else:
                        ouot[21] = min(ouot[20],abs(diff_x))
    
            if diff_x == -diff_y:
                if diff_x < 0:
                    if ouot[22] == 0:
                        ouot[22] = abs(diff_x)
                    else:
                        ouot[22] = min(ouot[22],abs(diff_x))
                else:
                    if ouot[23] == 0:
                        ouot[23] = abs(diff_x)
                    else:
                        ouot[23] = min(ouot[23],abs(diff_x))
                    
    
        for i in range(len(ouot)):
            if ouot[i] != 0:
                ouot[i] = (self.size - ouot[i] - 1) / (self.size - 2)
    
        return out.flatten()
