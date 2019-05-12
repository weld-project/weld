# Serialization in Weld

Weld supports serialization of its data types. The language contains two operators to acheive this:

* `serialize(data)` takes a value and returns a `vec[u8]`. This vector's data pointer contains a buffer with no
pointers, so it can be written to disk or sent across the network. The length of the vector designates the number
of bytes in the serialized buffer.
* `deserialize[T](data)` takes as input a value of type `vec[u8]` and returns a Weld value of type `T`.

Weld supports serialization of structs, vectors, scalars, and dictionaries, and
*does not* support serialization of SIMD values and builders.

### Serialization Formats

Weld serializes each data type in a determinstic way. The serialization format
currently does not encode the type in the serialized buffer, so users who call
`deserialize` must know the expected output type. This may be amended in a
later release. Currently, passing an incorrect type to `deserialize` will
result in undefined behavior, though if exactly every byte of the input
`vec[u8]` is not consumed exactly by `deserialize`, the runtime will abort with
a `DeserializationError`. Note that this check occurs after Weld attempts to
deserialize the input buffer, so this does not prevent all unsafe
behavior/protect against corrupt data. Each type in Weld is serialized as follows.

#### Scalars

A scalar is serialized as itself in a packed format. `i1` and `i8` both become
one-byte values, `i32` and `f32` and their unsigned counterparts (if
applicable) become 4-byte values, and `i64` and `f64` and their unsigned
counterparts become  8-byte values.

#### Structs

Structs are encoded as packed structs. The type `{T1,T2,..Tn}` becomes

```
[ serialize(T1) ] ... [ serialize(Tn) ]
```

#### Vectors

A vector is serialized as:

```
[8-byte length] [ serialize(element1) ] ... [ serialize(elementN) ]
```

#### Dictionaries

A dictionaries is serialized as:

```
[8-byte length (in key-value pairs)] [ serialize(key1) ] [ serialize(value1) ] ... [ serialize(keyN) ] [ serialize(valueN) ]
```

### Implementation Notes

The implementation currently optimizes data types that do not contain pointers by performing a fast
`memcpy` operation instead of iterating over collections and serializing each one individually.
