#
# Configuration for the Black Scholes workload.
#
import numpy as np

# SIZE = (2 << 28)
SIZE = (2 << 25)

def get_data():
    # random prices between 1 and 101
    price = np.float32(np.random.rand(SIZE) * 100)
    # random prices between 0 and 101
    strike = np.float32(np.random.rand(SIZE) * 100)
    # random maturity between 0 and 4
    t = np.float32(1 + np.random.rand(SIZE) * 6)
    # random rate between 0 and 1
    rate = np.float32(0.01 + np.random.rand(SIZE))
    # random volatility between 0 and 1
    vol = np.float32(0.01 + np.random.rand(SIZE))

    return price, strike, t, rate, vol
