import numpy as np

ran = np.random.randint(1e4,size=1)[0]

def seed_func():
    np.random.seed(1)


def second_func():
    np.random.seed(ran)
    print(np.random.randint(1e4,size=1))
    
seed_func()
second_func()
