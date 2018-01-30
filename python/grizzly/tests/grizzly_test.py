#!/usr/bin/python

import grizzly.grizzly as gr
import pandas as pd
import numpy as np
import unittest


class PandasWeldTestMethods(unittest.TestCase):
    # TODO: Add more tests here

    def test_groupby_sum(self):
        # test single column
        df = pd.DataFrame({"a":[3,2,3], "b":[4,5,6]})
        input = gr.DataFrameWeld(df)
        groupby = input.groupby("a").sum().evaluate(False)
        self.assertItemsEqual([5, 10], groupby.to_pandas()["b"])

        # test multi column
        df = pd.DataFrame({"a":[3,2,3], "b":[4,5,6], "c":[6, 4, 3]})
        input = gr.DataFrameWeld(df)
        groupby = input.groupby("a").sum().evaluate(False)
        self.assertItemsEqual([5, 10], groupby.to_pandas()["b"])
        self.assertItemsEqual([4, 9], groupby.to_pandas()["c"])

        # test multikey single column
        df = pd.DataFrame({"a":[3,2,3], "b":[2,3,2], "c":[6, 5, 4]})
        input = gr.DataFrameWeld(df)
        groupby = input.groupby(["a", "b"]).sum().evaluate(False)
        self.assertItemsEqual([5, 10], groupby.to_pandas()["c"])

        # test multikey multi column
        df = pd.DataFrame({"a":[3,2,3], "b":[2,3,2], "c":[6, 5, 4], "d":[6, 4, 3]})
        input = gr.DataFrameWeld(df)
        groupby = input.groupby(["a", "b"]).sum().evaluate(False)
        self.assertItemsEqual([5, 10], groupby.to_pandas()["c"])

    def test_filter_self(self):
        k = np.array(["A", "b", "D", "F", "e", "A", "A", "ac"], dtype=str)
        s = np.array(["A", "b", "D", "F", "e", "A", "A", "ac"], dtype=str)
        df = pd.DataFrame({"k":k, "s":s})
        grs = gr.DataFrameWeld(df)
        grs = grs["k"].unique()

    def test_filter_self(self):
        input = gr.SeriesWeld(np.array(["False", "True", "True", "False"], dtype=str),gr.WeldVec(gr.WeldChar()))
        print input.lower().evaluate()
        print input.filter(input.lower() == "true").evaluate()

    def test_groupby_sort(self):
        # test single column
        df = pd.DataFrame({"a":[3,2,3,2, 3, 2, 3, 2], "b":[6,7,4,5,4,2,1,7], "c":[4,5,6,7,8,9,1,4]})

        input = gr.DataFrameWeld(df)
        groupby = input.groupby("a").apply(lambda g: g.sort_values(by='b').slice(0, 2))
        groupby.reset_index()
        groupby = groupby.evaluate().to_pandas()
        mdf = df.groupby("a").apply(lambda g: g.sort_values(by='b')[0:2])
        mdf.reset_index(inplace=True, drop=True)
        self.assertItemsEqual(groupby, mdf)

    def test_unique(self):
        inp = gr.SeriesWeld(
            np.array(["aaa", "bbb", "aaa", "ccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        self.assertItemsEqual(["aaa", "bbb", "ccc"],
                              inp.unique().evaluate(False))

    def test_sum(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 3, 4, 5], dtype=np.int32), gr.WeldInt())
        self.assertEqual(15, inp.sum().evaluate(False))

    def test_count(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 3, 4, 5], dtype=np.int32), gr.WeldInt())
        self.assertEqual(5, inp.count().evaluate(False))

    def test_eq(self):
        inp = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        expected_output = [True, False, False, False]
        self.assertSequenceEqual(
            expected_output, list(
                (inp == "aaaa").evaluate(False)))

    def test_simpler_eq(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 100, 1, 2, 65], dtype=np.int32), gr.WeldInt())
        expected_output = [False, False, True, False, False, False]
        self.assertSequenceEqual(
            expected_output, list(
                (inp == 100).evaluate(False)))

    def test_mask(self):
        inp = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        predicates = np.array([True, False, False, True], dtype=bool)
        expected_output = ["bbbb", "bbb", "aa", "bbbb"]
        self.assertSequenceEqual(
            expected_output,
            list(
                inp.mask(
                    predicates,
                    "bbbb").evaluate(False)))

    def test_filter(self):
        inp = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        predicates = np.array([True, False, False, True], dtype=bool)
        expected_output = ["aaaa", "cccc"]
        self.assertSequenceEqual(expected_output, list(
            inp.filter(predicates).evaluate(False)))

    def test_eq_and_mask(self):
        inp = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        predicates = inp == "bbb"
        expected_output = ["aaaa", "eeee", "aa", "cccc"]
        self.assertSequenceEqual(
            expected_output,
            list(
                inp.mask(
                    predicates,
                    "eeee").evaluate(False)))

    def test_eq_and_mask_grouped(self):
        inp = gr.SeriesWeld(
            np.array([4, 3, 2, 1, 3], dtype=np.int32),
            gr.WeldInt())
        predicates = inp == 3
        output = gr.group(
            [predicates,
             inp.mask(predicates, 6)]
        ).evaluate(False)
        self.assertSequenceEqual([False, True, False, False, True], list(output[0]))
        self.assertEqual([4, 6, 2, 1, 6], list(output[1]))

    def test_eq_and_sum_grouped(self):
        inp = gr.SeriesWeld(
            np.array([4, 3, 2, 1, 3], dtype=np.int32),
            gr.WeldInt())
        predicates = inp == 3
        output = gr.group(
            [predicates,
             inp.sum()]
        ).evaluate(False)
        self.assertSequenceEqual([False, True, False, False, True], list(output[0]))
        self.assertEqual(13, output[1])

    def test_eq_and_mask_grouped_strings(self):
        inp = gr.SeriesWeld(
            np.array(["aaaa", "bbb", "aa", "cccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        predicates = inp == "bbb"
        expected_output = ["aaaa", "eeee", "aa", "cccc"]
        output = gr.group(
            [predicates,
             inp.mask(predicates, "eeee")]
        ).evaluate(False)
        self.assertSequenceEqual([False, True, False, False], list(output[0]))
        self.assertSequenceEqual(["aaaa", "eeee", "aa", "cccc"], list(output[1]))

    def test_slice(self):
        inp = gr.SeriesWeld(
            np.array(["aaaaaa", "bbbbbb", "aaaaaaa", "cccccccc"], dtype=str),
            gr.WeldVec(gr.WeldChar()))
        expected_output = ["aaa", "bbb", "aaa", "ccc"]
        self.assertSequenceEqual(
            expected_output, list(
                inp.str.slice(
                    1, 3).evaluate(False)))

    def test_add(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [2, 4, 6, 8]
        self.assertSequenceEqual(
            expected_output, list(
                inp.add(other).evaluate(False)))

    def test_sub(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [0, 0, 0, 0]
        self.assertSequenceEqual(
            expected_output, list(
                inp.sub(other).evaluate(False)))

    def test_mul(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [1, 4, 9, 16]
        self.assertSequenceEqual(
            expected_output, list(
                inp.mul(other).evaluate(False)))

    def test_div(self):
        inp = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        other = gr.SeriesWeld(
            np.array([1, 2, 3, 4], dtype=np.int32), gr.WeldInt())
        expected_output = [1, 1, 1, 1]
        self.assertSequenceEqual(
            expected_output, list(
                inp.div(other).evaluate(False)))

if __name__ == '__main__':
    unittest.main()
