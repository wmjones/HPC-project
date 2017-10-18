import multiprocessing


class Config:
    # AGENTS = multiprocessing.cpu_count()
    AGENTS = 4
    PREDICTORS = 1
    TRAINERS = 1
    DEVICE = 'cpu:0'

    PREDICTION_BATCH_SIZE = 10
    LEARNING_RATE = 1e-3
    TRAINING_MIN_BATCH_SIZE = 100
    RESTORE = False
