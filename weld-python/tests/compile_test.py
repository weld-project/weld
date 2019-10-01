"""
Ad-hoc test for compilation.

This needs to be integrated with the pytest test suite.

"""

import weld
import weld.compile
import weld.types

# Single argument program.
program = "|x: i32| x + 1"
add_one = weld.compile.compile(program, [weld.types.I32()], [None], weld.types.I32(), None) 

print(add_one(1))
print(add_one(5))


inner = weld.types.WeldStruct((weld.types.I32(), weld.types.I32()))
outer = weld.types.WeldStruct((weld.types.I32(), inner))
print(outer)

program = "|x: i32, y: i32| {x + y, {1, 1}}"
add = weld.compile.compile(program,
        [weld.types.I32(), weld.types.I32()],
        [None, None],
        outer,
        None) 

print(add(5, 5))
