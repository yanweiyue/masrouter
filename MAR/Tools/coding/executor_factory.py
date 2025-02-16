#!/usr/bin/env python
# -*- coding: utf-8 -*-


from MAR.Utils.log import logger
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Tools.coding.executor_types import Executor

EXECUTOR_MAPPING = {
    "py": PyExecutor,
    "python": PyExecutor,
}

def executor_factory(lang: str) -> Executor:

    if lang not in EXECUTOR_MAPPING:
        raise ValueError(f"Invalid language for executor: {lang}")

    executor_class = EXECUTOR_MAPPING[lang]
    return executor_class()