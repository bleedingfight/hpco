import numpy as np
with open('/tmp/softmax_input.txt', 'r') as f:
    content = [eval(i) for i in f.read().strip().split(' ')]
first_line = np.array(content[:1024])
warp_size = 32
num_warps = 1024 // warp_size
for i in range(num_warps):
    warp = first_line[i * warp_size:(i + 1) * warp_size]
    max_val = np.max(warp)
    den = np.sum(np.exp(warp - max_val))
    print("index = {} Warp {}: den Output: {}".format(i, max_val, den))
