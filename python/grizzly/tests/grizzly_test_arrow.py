#!/usr/bin/python

import grizzly.grizzly as gr
import numpy as np
import pyarrow as pa
import unittest
from functools import partial
from grizzly import grizzly_multi_impl

ArrowArray = partial(gr.SeriesWeld, impl=grizzly_multi_impl.grizzly_arrow_impl)

def nd2pa(arr):
    return pa.Array.from_pandas(arr)


class ArrowWeldTestMethods(unittest.TestCase):
    # TODO: Add more tests here

    def test_sum(self):
        inp = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4, 5], dtype=np.int32)),
            gr.WeldInt())
        self.assertEqual(15, inp.sum().evaluate(False))

    def test_count(self):
        input = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4, 5], dtype=np.int32)),
            gr.WeldInt())
        self.assertEqual(5, input.count().evaluate(False))


    def test_simpler_eq(self):
        input = ArrowArray(
            nd2pa(np.array([1, 2, 100, 1, 2, 65], dtype=np.int32)),
            gr.WeldInt())
        output = (input == 100).evaluate(False)
        expected_output = [False, False, True, False, False, False]
        self.assertSequenceEqual(
            expected_output, list(
                (input == 100).evaluate(False).to_pandas()))

    def test_add(self):
        input = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        other = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())

        output = list(input.add(other).evaluate(False).to_pandas())
        expected_output = list(np.array([2, 4, 6, 8], dtype=np.int32))
        self.assertSequenceEqual(
            expected_output, output)

    def test_sub(self):
        input = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        other = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        expected_output = [0, 0, 0, 0]
        self.assertSequenceEqual(
            expected_output, list(
                input.sub(other).evaluate(False).to_pandas()))

    def test_mul(self):
        input = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        other = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        expected_output = [1, 4, 9, 16]
        self.assertSequenceEqual(
            expected_output, list(
                input.mul(other).evaluate(False).to_pandas()))

    def test_div(self):
        input = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        other = ArrowArray(
            nd2pa(np.array([1, 2, 3, 4], dtype=np.int32)),
            gr.WeldInt())
        expected_output = [1, 1, 1, 1]
        self.assertSequenceEqual(
            expected_output, list(
                input.div(other).evaluate(False).to_pandas()))


if __name__ == '__main__':
    unittest.main()
