#!/usr/bin/python

import grizzly.numpy_weld as npw
import numpy as np
import unittest


class NumPyWeldTestMethods(unittest.TestCase):
    # TODO: Add more tests here

    def test_div(self):
        input = npw.NumpyArrayWeld(
            np.array([3, 6, 12, 15, 14], dtype=np.int32), npw.WeldInt())
        self.assertItemsEqual([1, 2, 4, 5, 4], (input / 3).evaluate(False))

    def test_sum(self):
        input = npw.NumpyArrayWeld(
            np.array([1, 2, 3, 4, 5], dtype=np.int32), npw.WeldInt())
        self.assertEqual(15, input.sum().evaluate(False))

    def test_dot(self):
        matrix = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int32)
        vector = np.array([0, 1, 2], dtype=np.int32)
        self.assertItemsEqual([8, 11, 14], npw.dot(
            matrix, vector).evaluate(False))


if __name__ == '__main__':
    unittest.main()
