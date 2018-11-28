#!usr/bin/python

import numpy as np

if __name__ == '__main__':
    nStep = 1000000
    position = 0
    position_record = [0]

    for i in range(nStep):
        choose = 1 if np.random.randint(0,2,1) else -1
        position += choose
        position_record.append(position)

    print('after %d steps, point arrive %d'%(nStep, position))