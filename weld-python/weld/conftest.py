
import pytest

import numpy as np
import pandas as pd
import weld.grizzly as gr

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    """
    Make `gr`, `np`,  and `pd available for doctests.
    """
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["gr"] = gr
