
import ctypes
import weld_python.weld_python as weld

conf = weld.WeldConf()
context = weld.WeldContext(conf)

module = weld.WeldModule("|x: i32| x + 10", conf)

pointer = ctypes.POINTER(ctypes.c_int)
b = ctypes.c_int(20)

value = weld.WeldValue(ctypes.addressof(b))
result = module.run(context, value)

data = result.data()
data = ctypes.cast(data, pointer)
print("Result:", data.contents)
print("Memory usage:", context.memory_usage(), "bytes")




