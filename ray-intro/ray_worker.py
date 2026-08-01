import ray
ray.init()


@ray.remote  # each actor = 1 process, gets 1 CPU by default
class Worker:
    def __init__(self, wid):
        self.wid = wid
        self.count = 0

    def do(self, x):
        self.count += 1
        return f"worker {self.wid}: {x*x} (call #{self.count})"


# spin up one instance per core
workers = [Worker.remote(i) for i in range(14)]

# fan work out across them
results = ray.get([w.do.remote(i) for i, w in enumerate(workers)])
print(results)

ray.shutdown()
