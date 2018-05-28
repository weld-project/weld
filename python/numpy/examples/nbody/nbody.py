from __future__ import print_function
import psutil
import os
"""
NBody in N^2 complexity

Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import numpy as np
import weldnumpy as wn
# import weldnumpy as np
from weldnumpy import weldarray
import argparse
import time

G     = np.float64(6.67384e-11)     # m/(kg*s^2)
dt    = np.float64(60*60*24*365.25) # Years in seconds
r_ly  = np.float64(9.4607e15)       # Lightyear in m
m_sol = np.float64(1.9891e30)       # Solar mass in kg

# with smaller values, parts of the computation, like the mask, does not quite work out.
# G     = np.float64(6.67384)     # m/(kg*s^2)
# dt    = np.float64(60.0) # Years in seconds
# r_ly  = np.float64(9.0)       # Lightyear in m
# m_sol = np.float64(1.9891)       # Solar mass in kg

def diagonal(ary, offset=0):
    """
    """
    if ary.ndim != 2:
        raise Exception("diagonal only supports 2 dimensions\n")
    if offset < 0:
        offset = -offset
        if (ary.shape[0]-offset) > ary.shape[1]:
            ary_diag = ary[offset, :]
        else:
            ary_diag = ary[offset:, 0]
    else:
        if ary.shape[1] - offset > ary.shape[0]:
            ary_diag = ary[:, offset]
        else:
            ary_diag = ary[0, offset:]

    ary_diag.strides = (ary.strides[0]+ary.strides[1],)

    if isinstance(ary_diag, weldarray):
        ary_diag._weldarray_view.strides = ary_diag.strides

    return ary_diag

def random_galaxy(N, dtype=np.float64):
    """Generate a galaxy of random bodies"""
    np.random.seed(1)
    galaxy = {            # We let all bodies stand still initially
        'm':    (np.random.rand(N,) + dtype(10)) * dtype(m_sol/10),
        'x':    (np.random.rand(N,) - dtype(0.5)) * dtype(r_ly/100),
        'y':    (np.random.rand(N,) - dtype(0.5)) * dtype(r_ly/100),
        'z':    (np.random.rand(N,) - dtype(0.5)) * dtype(r_ly/100),
        'vx':   np.zeros(N, dtype=dtype),
        'vy':   np.zeros(N, dtype=dtype),
        'vz':   np.zeros(N, dtype=dtype)
    }

    if dtype == np.float32:
        galaxy['m'] /= 1e10
        galaxy['x'] /= 1e5
        galaxy['y'] /= 1e5
        galaxy['z'] /= 1e5

    return galaxy

def move(galaxy, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """
    np.random.seed(1)
    # Calculate all distances component wise (with sign)
    start = time.time()

    dx = np.transpose(galaxy['x'][np.newaxis,:]) - galaxy['x']
    dy = np.transpose(galaxy['y'][np.newaxis,:]) - galaxy['y']
    dz = np.transpose(galaxy['z'][np.newaxis,:]) - galaxy['z']

    # Euclidian distances (all bodys)
    # temporary: until we fix power operator.
    #r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.sqrt(dx*dx + dy*dy + dz*dz)

    diagonal(r)[:] = 1.0

    # if isinstance(r, weldarray):
    if False:
        # stupid hack because cmp ops behave differently in numpy --> returns boolean arrays, and
        # multiplying that with f64's would not work for us.
        mask = r._cmp_op(1.0, np.less.__name__)
        not_mask = r._cmp_op(1.0, np.greater_equal.__name__)
    else:
	print('using numpy version for mask!')
        mask = r < 1.0
        not_mask = r >= 1.0

    # this will get offloaded too.
    r = (r * not_mask) + mask

    # TODO: fix power.
    # r2 = r**3
    r2 = r*r*r

    m = np.transpose(galaxy['m'][np.newaxis,:])

    Fx = (G*dx*m)/r2
    Fy = (G*dy*m)/r2
    Fz = (G*dz*m)/r2

    # Set the force (acceleration) a body exerts on it self to zero
    diagonal(Fx)[:] = 0.0
    diagonal(Fy)[:] = 0.0
    diagonal(Fz)[:] = 0.0

    galaxy['vx'] += dt*np.sum(Fx, axis=0)
    galaxy['vy'] += dt*np.sum(Fy, axis=0)
    galaxy['vz'] += dt*np.sum(Fz, axis=0)

    galaxy['x'] += dt*galaxy['vx']
    galaxy['y'] += dt*galaxy['vy']
    galaxy['z'] += dt*galaxy['vz']

    # what we return is irrelevant to nbody computations, but we just use it
    # for sanity check when comparing with numpy.
    return Fx

def simulate(galaxy, timesteps, visualize=False):
    for i in range(timesteps):
        ret = move(galaxy,dt)
        for k, v in galaxy.iteritems():
            if isinstance(v, weldarray):
                galaxy[k] = v.evaluate()
    return ret

def compare(R, R2):

    if isinstance(R2, weldarray):
        R2 = R2.evaluate()

    mistakes = 0
    R = R.flatten()
    R2 = R2.view(np.ndarray).flatten()

    assert R.dtype == R2.dtype, 'dtypes must match!'

    # this loop takes ages to run so just avoiding it.
    # for i, r in enumerate(R):
        # # if not np.isclose(R[i], R2[i]):
        # # if we use exact match, then sometimes things seem to be different many decimal places
        # down...
        # if R[i] != R2[i]:
            # mistakes += 1
            # print('R[i] : ', R[i])
            # print('R2[i]: ', R2[i])
    # if mistakes != 0:
        # print('mistakes % = ', mistakes / float(len(R)))
    # else:
        # print('mistakes = 0')

    assert np.allclose(R, R2)
    print('comparison successful!')

def print_args(args):
    d = vars(args)
    print('params: ', str(d))

def main():

    parser = argparse.ArgumentParser(
        description="give num_els of arrays used for nbody"
    )
    parser.add_argument('-n', "--num_els", type=int, required=True,
                        help="num_els of 1d arrays")
    parser.add_argument('-t', "--num_times", type=int, required=True,
                        help="num iterations for loop")
    parser.add_argument('-p', "--remove_pass", type=str,
                        default="whatever_string", help="will remove the pass containing this str")
    parser.add_argument('-numpy', "--use_numpy", type=int, required=False, default=0,
                        help="use numpy or not in this run")
    parser.add_argument('-weld', "--use_weld", type=int, required=False, default=0,
                        help="use weld or not in this run")

    args = parser.parse_args()
    print_args(args)

    if args.use_numpy:
        galaxy = random_galaxy(args.num_els)
        # First run numpy
        start = time.time()
        ret1 = simulate(galaxy, args.num_times)
        R = galaxy['x'] + galaxy['y'] + galaxy['z']
        # assert R.dtype == np.float32, 'should be float64'
        end = time.time()
        print('****************************')
        print('numpy took {} seconds'.format(end-start))
        print('****************************')
    else:
        print('Not running numpy')

    if args.use_weld:
        # Part 2: Weld.
        galaxy2 = random_galaxy(args.num_els)
        for k, v in galaxy2.iteritems():
            galaxy2[k] = weldarray(v)

        start = time.time()
        ret2 = simulate(galaxy2, args.num_times)

        R2 = galaxy2['x'] + galaxy2['y'] + galaxy2['z']
        if isinstance(R2, weldarray):
            R2 = R2.evaluate()

        end = time.time()
        print('****************************')
        print('weld took {} seconds'.format(end-start))
        print('****************************')
    else:
        print('Not running weld')

    if args.use_numpy and args.use_weld:
        compare(ret1, ret2)
        compare(R, R2)

if __name__ == "__main__":
    main()
