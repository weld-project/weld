#!/usr/bin/python

import grizzly.grizzly as gr
import numpy as np
import unittest


class PandasWeldTestMethods(unittest.TestCase):
    # TODO: Add more tests here

    def test_unique(self):
        input = gr.SeriesWeld(
            np.array(["aaa", "bbb", "aaa", "ccc"], dtype=str), gr.WeldVec(gr.WeldChar()))
        self.assertItemsEqual(["aaa", "bbb", "ccc"],
                              input.unique().evaluate(False))

    def test_sum(self):
        inp = gr.SeriesWeld(np.array([1, 2, 3, 4, 5], dtype=np.int32), gr.WeldInt())
        self.assertEqual(15, inp.sum().evaluate(False))

    def test_count(self):
        input = gr.SeriesWeld(np.array([1, 2, 3, 4, 5], dtype=np.int32), gr.WeldInt())
        self.assertEqual(5, input.count().evaluate(False))

    def test_eq(self):
        input = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str), gr.WeldVec(gr.WeldChar()))
        expected_output = [True, False, False, False]
        self.assertSequenceEqual(
            expected_output, list(
                (input == "aaaa").evaluate(False)))

    def test_simpler_eq(self):
        input = gr.SeriesWeld(
            np.array([1, 2, 100, 1, 2, 65], dtype=np.int32), gr.WeldInt())
        expected_output = [False, False, True, False, False, False]
        self.assertSequenceEqual(
            expected_output, list(
                (input == 100).evaluate(False)))

    def test_mask(self):
        input = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str), gr.WeldVec(gr.WeldChar()))
        predicates = np.array([True, False, False, True], dtype=bool)
        expected_output = ["bbbb", "bbb", "aa", "bbbb"]
        self.assertSequenceEqual(
            expected_output,
            list(
                input.mask(
                    predicates,
                    "bbbb").evaluate(False)))

    def test_filter(self):
        input = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str), gr.WeldVec(gr.WeldChar()))
        predicates = np.array([True, False, False, True], dtype=bool)
        expected_output = ["aaaa", "cccc"]
        self.assertSequenceEqual(expected_output, list(
            input.filter(predicates).evaluate(False)))

    def test_eq_and_mask(self):
        input = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str), gr.WeldVec(gr.WeldChar()))
        predicates = input == "bbb"
        expected_output = ["aaaa", "eeee", "aa", "cccc"]
        self.assertSequenceEqual(
            expected_output,
            list(
                input.mask(
                    predicates,
                    "eeee").evaluate(False)))

    def test_slice(self):
        input = gr.SeriesWeld(
            np.array(["aaaaaa", "bbbbbb", "aaaaaaa", "cccccccc"], dtype=str), gr.WeldVec(gr.WeldChar()))
        expected_output = ["aaa", "bbb", "aaa", "ccc"]
        self.assertSequenceEqual(
            expected_output, list(
                input.str.slice(
                    1, 3).evaluate(False)))

    def test_add(self):
        input = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [2, 4, 6, 8]
        self.assertSequenceEqual(
            expected_output, list(
                input.add(other).evaluate(False)))

    def test_sub(self):
        input = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [0, 0, 0, 0]
        self.assertSequenceEqual(
            expected_output, list(
                input.sub(other).evaluate(False)))

    def test_mul(self):
        input = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [1, 4, 9, 16]
        self.assertSequenceEqual(
            expected_output, list(
                input.mul(other).evaluate(False)))

    def test_div(self):
        input = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [1, 1, 1, 1]
        self.assertSequenceEqual(
            expected_output, list(
                input.div(other).evaluate(False)))

if __name__ == '__main__':
    unittest.main()
