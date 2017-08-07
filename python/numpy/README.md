Directory Structure:

  - weldarray.py: Weld Array file, which subclasses ndarray from numpy.
  - weldnumpy.py: Helper functions, should eventually have stuff like np.zeros
  etc.

  ./tests: contains all the tests.
  Run with the command 'pytest'.

  ./timings:

    - timings.py: long loop with multiple ops being tested with (numpy,
        weldarray)

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
    - So far I'm going with subclassing.

  - Views:
    - dealt with in __getitem__, along with other indexing strategies [..]
    - Always evaluate registered ops, and then let ndarray superclass create the view.
      - The new view shares underlying memory with parent if possible (if
          ndarray can avoid copying the stuff). Since we aren't making changes
      to the underlying memory - but storing future ops - we need to deal with
      it separately.
      - Keep a child list / parent list for any view/parent - and whenever
      updating these - update the elements in this list too. This seems a
      little awkward...but I didn't see a nicer alternative.    

