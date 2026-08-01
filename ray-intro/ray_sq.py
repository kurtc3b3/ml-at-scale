import ray

ray.init()  # starts a local cluster on this machine


@ray.remote
def square(x):
    return x * x


futures = [square.remote(i) for i in range(4)]
print(ray.get(futures))  # [0, 1, 4, 9]

ray.shutdown()
