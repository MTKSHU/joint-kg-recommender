import multiprocessing

class MyProcess(multiprocessing.Process):
	def __init__(self, L, queue=None):
		super(MyProcess, self).__init__()
		self.L = L
		self.queue = queue

	def run(self):
		while True:
			x = self.queue.get()
			try:
				self.process_data(x, self.L)
			except:
				time.sleep(5)
				self.process_data(x, self.L)
			self.queue.task_done()

	def process_data(self, x, L):
		L.append((x, x**2))

inputs = [i for i in range(10)]

with multiprocessing.Manager() as manager:
    L = manager.list()
    queue = multiprocessing.JoinableQueue()
    workerList = []
    for i in range(5):
        worker = MyProcess(L, queue=queue)
        workerList.append(worker)
        worker.daemon = True
        worker.start()

    for x in inputs:
        queue.put(x)

    queue.join()

    resultList = list(L)

    for worker in workerList:
        worker.terminate()

    print(resultList)