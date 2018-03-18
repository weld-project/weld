## Serialization

Various tests and examples for serializing data in Weld.

* `deserialize_vec_pointers/` serializes a vector, and then deserializes. The vector has pointers in it, which
the serialization flattens.

* `serialize_vec/` serializes a vector of structs with no pointers.

* `serialize_dictionary/` serializes and deserializes a dictionary with and without pointers.
