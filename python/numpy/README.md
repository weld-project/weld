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

Design Decisions: (see google doc for details)

  - NdArray {binop} WeldArray --> WeldArray
    - First evaluate all the lazily registered ops (as the function calling it
        might want to use the return value for some computation, e.g.,
        np.allclose)
    - Return a WeldArray (as long as types allow)

  - Sharing underlying memory when generating new WeldArrays:
    - e.g., a = np.add(b, c).
    - ‘a’, and ‘b’ share the underlying ndarrays, but their weldobjects are different - so they can essentially be treated as separate ndarrays.

  - Implicit Evaluation: This might be the desired behaviour whenever the user tries to access the array in some way. For instance:

    Proposed Solution: After every eval() --> just generate a new weldobject
    with the returned ndarray. Thus, future calls to eval() will not need to
    repeat previous calculation, and will have access to the latest results.
    This essentially is equivalent to changing the data pointer to the np array
    as we will never access the original np memory for this array.

  - Subclassing ndarray vs Having a separate object:
    - So far I'm going with subclassing, but while implementing some of the
    things, particularly the above point - I see some good reasons why a
    separate object might be better.  Subclassing Pros:


  - Views:
    Just do the following:
    - Evaluate the base array (so all previously registered ops are done). Will get a ndarray back.
    - Find the relevant indices, index into this array, and create a new weldobject with this array as the base parameter.
   - Should work the same as other cases after.

