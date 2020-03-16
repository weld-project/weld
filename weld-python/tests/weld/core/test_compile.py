"""
Tests for compilation.
"""

import pytest

import weld
import weld.compile
import weld.types

def test_simple():
    # Single argument program.
    program = "|x: i32| x + 1"
    add_one = weld.compile.compile(program,
            [weld.types.I32()],
            [None],
            weld.types.I32(),
            None) 
    assert add_one(1)[0] == 2
    assert add_one(5)[0] == 6
    assert add_one(-4)[0] == -3

def test_exception():
    with pytest.raises(weld.WeldError):
        # Single argument program.
        program = "|x: i32| x + ERROR"
        program = weld.compile.compile(program,
                [weld.types.I32()],
                [None],
                weld.types.I32(),
                None) 

def test_nested():
    inner = weld.types.WeldStruct((weld.types.I32(), weld.types.I32()))
    outer = weld.types.WeldStruct((weld.types.I32(), inner))
    program = "|x: i32, y: i32| {x + y, {1, 1}}"
    add = weld.compile.compile(program,
        [weld.types.I32(), weld.types.I32()],
        [None, None],
        outer,
        None) 
    assert add(5, 5)[0] == (10, (1, 1))
