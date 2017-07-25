Directory Structure:

  - weld_numpy_elemwise.py: Weld Numpy file which uses a separate object. Was
  using this for timing comparisons with the numpy subclass file - but this
  hasn't been updated since before.

  - weldarray.py: is the new Weld Numpy file, which subclasses from numpy.
  Mostly, it extends on weld_numpy_elemwise.

  ./tests:
  The tests were written for the numpy subclass weldarray.py, and can be run
  with the command 'pytest' from the base directory.

  ./timings:

    - timings.py: long loop with multiple ops being tested with (numpy,
        weldarray (subclass), weld array as a separate object).

    - blackscholes stuff: Various files that implement the blackscholes
    workload - need to change these to get it in the same form as timings.py.
