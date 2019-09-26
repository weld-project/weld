import weld
import weld.compile
import weld.types

# Single argument program.
program = "|x: i32| x + 1"
add_one = weld.compile.compile(program, [weld.types.I32()], [None], weld.types.I32(), None) 

print(add_one(1))
print(add_one(5))

program = "|x: i32, y: i32| x + y"
add = weld.compile.compile(program,
        [weld.types.I32(), weld.types.I32()],
        [None, None],
        weld.types.I32(),
        None) 

print(add(1, 1))
print(add(5, 5))
