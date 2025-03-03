import math
import random

class Machine:
    def __init__(self, id, expected_service_time, buffer_size = None):
        self.id = id
        self.next = None
        self.prev = None
        self.expected_service_time = expected_service_time
        self.completion_time = float('inf')
        self.busy = False
        self.completed_items = 0

        if buffer_size != None:
            self.buffer = Buffer(buffer_size)
        else: self.buffer = None
    
    def __str__(self):
        return f'{self.id}'
    
    def resetCounter(self):
        self.completed_items = 0
    
    def addNext(self, machine):
        self.next = machine
        machine.prev = self

    def getServiceTime(self):
        return self.expected_service_time

    def hasBuffer(self):
        return self.buffer != None
    
    def nextBufferNotFull(self):
        return not self.next or not self.next.buffer.isFull()
    
    def hasItemToProcess(self):
        return not self.hasBuffer() or not self.buffer.isEmpty()
    
    def prevBufferNotEmpty(self):
        return self.prev and (not self.prev.hasBuffer() or not self.prev.buffer.isEmpty())

    def startService(self, current_time):
        # Remove item from buffer
        if self.hasBuffer():
            self.buffer.removeItem()

        # Set completion time and status (idle or busy)
        self.completion_time = current_time + self.getServiceTime()
        self.busy = True

        # See if the previous machine was idle because the buffer was full
        if self.prevBufferNotEmpty() and not self.prev.busy:
            self.prev.startService(current_time)
        
    def completeService(self):
        # Get time item was completed
        time_completion = self.completion_time

        # Push item to the next buffer
        if self.next and not self.next.buffer.isFull():
            self.next.buffer.addItem()

            # If idle and next buffer is not full, immediately start service 
            if not self.next.busy and (not self.next.next or not self.next.next.buffer.isFull()):
                self.next.startService(self.completion_time)
        
        # If possible start service of new item
        if self.nextBufferNotFull() and self.hasItemToProcess():
            self.startService(self.completion_time)

        else: 
            self.busy = False
            self.completion_time = float('inf')

        self.completed_items += 1

        return time_completion

class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size + 1
        self.current_size = 0

    def __str__(self):
        return f'{self.current_size}'
    
    def isFull(self):
        return self.current_size == self.max_size
    
    def isEmpty(self):
        return self.current_size == 0
    
    def addItem(self):
        self.current_size += 1

    def removeItem(self):
        self.current_size -= 1

class ExponentialMachine(Machine):
    def __init__(self, id, mu, buffer_size=None):
        super().__init__(id, 1 / mu, buffer_size)
        self.mu = mu
    
    def getServiceTime(self):
        return -math.log(1.0 - random.random()) / self.mu

def find_machine_first_completed(first_machine):
    completion_time = first_machine.completion_time
    machine = first_machine

    current_machine = first_machine

    while current_machine.next:
        current_machine = current_machine.next

        if current_machine.completion_time < completion_time:
            completion_time = current_machine.completion_time
            machine = current_machine

    return machine

def print_completion_times(first_machine):
    completion_times = [first_machine.completion_time]
    current_machine = first_machine

    while current_machine.next:
        current_machine = current_machine.next
        completion_times.append(current_machine.completion_time)

    print(completion_times)

def create_machine_list(mus, max_buffer_sizes):
    num_machines = len(mus)

    first_machine = ExponentialMachine(1, mus[0])
    prev_machine = first_machine

    for m in range(num_machines - 1):
        machine = ExponentialMachine(m + 2, mus[m + 1], max_buffer_sizes[m])

        prev_machine.addNext(machine)
        prev_machine = machine
    
    return first_machine

def run_loop(machines, runtime, start_time):
    current_time = start_time

    while current_time < runtime:
        machine = find_machine_first_completed(machines)
        current_time = machine.completeService()

def run_sim_exponential(mus, max_buffer_sizes, max_runtime, warmup_time):
    first_machine = create_machine_list(mus, max_buffer_sizes)
    first_machine.startService(0)

    run_loop(first_machine, warmup_time, 0)
    first_machine.next.next.resetCounter()
    print(first_machine.next.next.completed_items)

    run_loop(first_machine, max_runtime + warmup_time, warmup_time)

    print(first_machine.next.next.completed_items)
    print(first_machine.next.next.completed_items / max_runtime)
    return first_machine

mus = [1, 1.1, 0.9]
max_buffer_sizes = [5, 5]

run_sim_exponential(mus, max_buffer_sizes, 100000, 10000)