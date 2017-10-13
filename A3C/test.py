from multiprocessing import Process, Queue, Value
import time, random
import threading


class ProcessAgent(Process):
    def __init__(self, id):
        super(ProcessAgent, self).__init__()
        self.id = id
        self.exit_flag = Value('i', 0)
        self.count = 0

    def run(self):
        while self.exit_flag.value == 0:
            self.count += 1


class Server:
    def __init__(self):
        self.agents = []

    def main(self):
        for id in range(0, 4):
            self.agents.append(ProcessAgent(id))
            self.agents[-1].start()

        time.sleep(10)
        for id in range(0, 4):
            self.agents[-1].exit_flag.value = True
            self.agents[-1].join()
            self.agents.pop()


Server().main()
