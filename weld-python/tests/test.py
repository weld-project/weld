
import ctypes
import weld

conf = weld.WeldConf()
context = weld.WeldContext(conf)

module = weld.WeldModule("|x: i32| x + 10", conf)

pointer = ctypes.POINTER(ctypes.c_int)
b = ctypes.c_int(20)

value = weld.WeldValue(ctypes.addressof(b))
result = module.run(context, value)

ty = module.return_type()
print(str(ty))

data = result.data()
data = ctypes.cast(data, pointer)
print("Result:", data.contents)
print("Memory usage:", context.memory_usage(), "bytes")


# Try to compile a module that fails.
try:
    module = weld.WeldModule("|x: i32| x + 10 + whoops", conf)
except weld.WeldError as e:
    print(e)


