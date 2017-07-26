Directory Structure:

  - weldarray.py: Weld Numpy file, which subclasses from numpy.
  - main.py: Just some random code for testing features in isolation.

  ./tests:
  Run with the command 'pytest'.

  ./timings:

    - timings.py: long loop with multiple ops being tested with (numpy,
        weldarray (subclass), weld array as a separate object).

    - blackscholes stuff: Various files that implement the blackscholes
    workload - need to change these to get it in the same form as timings.py.

Design Decisions:

  - NdArray op WeldArray --> WeldArray
    - First evaluate all the lazy registered ops (as the function calling it
        might want to use the return value for some computation, e.g.,
        np.allclose)
    - Return a WeldArray (as long as types allow)

  - Sharing underlying memory when generating new WeldArrays:
    - e.g., a = np.add(b, c)
    - Since the ops aren't evaluated immediately, it makes sense to just update
    the weld IR code and not copy the elements of 'b'. But...
    - At this point, the user expects 'a', and 'b', to be different arrays with
    numpy / ndarray. So for semantic correctness, when we register the addition
    operation on 'b', we will have to copy the array before returning 'a'. This
    clearly sucks - causes a fairly big slowdown (in timings/timing.py, with
        reps = 100,000, weld timing went from 5 seconds to 10+ seconds, in
        comparison np is at 8 seconds)
    - This might be particularly annoying when it happens in some library code
    and the user doesn't realize it.
    - Possible way around it: avoid the copying - by maintaining extra
    information about which WeldArray it is, and sharing the underlying ndarray
    data...but lead to other edge cases.
    - Just ignore the issue for now?!

  - Implicit Evaluation: This might be the desired behaviour whenever the user
  tries to access the array in some way. For instance:
    - Whenever printing WeldArray
    - Whenever an unsupported op is called - then before calling numpy's
    implementation, evaluate all the ops stored so far.
    - In particular function paths, for instance, np.array_equal.
    - When indexing

    - Problems: Note we can't evaluate the array in place (for instance, in case of
        multi-dim arrays, with matrix ops).

      The main problem appears to be if it happens in some library call, for
  e.g., (np.isfinite in np.allclose) - and the external user might not be aware
  that eval has been called on his array (or might not be able to save the
      returned array)

      - If we do not store the result of the evaluation, will waste running the
      function in Weld multiple times. But this is safe.

      - If we store the value after evaluation, then will need to either store
      a new field in the WeldArray or modify the underlying class (maybe just
          change __data__ pointer?)

  - Subclassing ndarray vs Having a separate object:
    - So far I'm going with subclassing, the main worries seem to be:
      - Is the overhead in function calls etc extra? (doesn't seem to be the
          case, but don't have the equivalent representation of a separate
          object yet to compare). In general the overhead in creating new
      arrays / and routing function calls through __numpy_ufunc__ is quite big
      compared to numpy (which seems to implement these in C...)

      - Can't just change one import line to run existing numpy programs. In
      particular, all the functions would still be np.add, np.exp etc. but will
      just need to separately have a class that supplies equivalents of
      np.array, np.zeros, np.concatenate etc. (there seemed to be some
          discussion in numpy forums about allowing overwrite of all numpy
          functions, but it doesn't seem to be supported at least now)

  - Views:
    - When eval called on a view, for correctness, we will need to first eval
    the base array. Will need a way to get the chain of arrays that share
    memory with a given view.

  - Other Edge Cases, including fancy indexing etc:

