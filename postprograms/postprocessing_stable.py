#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 15:41:27 2015

@author: Satoshi
"""
from logging import getLogger, StreamHandler, Formatter, INFO
import output_stable
import graphs_stable


def postprocess_clean():
    # ロガーの設定
    logger = getLogger(__name__)
    shandler = StreamHandler()
    shandler.setFormatter(Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    shandler.setLevel(INFO)
    logger.setLevel(INFO)
    logger.addHandler(shandler)

    logger.info("output_ref")
    output_stable.ref()
    logger.info("output_predata")
    output_stable.pre()
    output_stable.data()
    logger.info("output_result")
    output_stable.result()
    logger.info("output_error")
    output_stable.error_drift()
    logger.info("result_graphs")
    graphs_stable.result()
    logger.info("offset_graphs")
    graphs_stable.offset()
    logger.info("rawdata_graphs")
    graphs_stable.rawdata()
    logger.info("error_graphs")
    graphs_stable.error()


def postprocess_fin():
    # ロガーの設定
    logger = getLogger(__name__)
    shandler = StreamHandler()
    shandler.setFormatter(Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    shandler.setLevel(INFO)
    logger.setLevel(INFO)
    logger.addHandler(shandler)

    logger.info("output_ref")
    output_stable.ref()
    logger.info("output_predata")
    output_stable.pre()
    output_stable.data()
    logger.info("output_result")
    output_stable.result(multiplesweep=True)
    # logger.info("output_error")
    # outputerror_drift.main()
    logger.info("result_graphs")
    graphs_stable.result_multi()
    logger.info("offset_graphs")
    graphs_stable.offset()
    # logger.info("rawdata_graphs")
    # rawdatagraphs.main()
    # logger.info("error_graphs")
    # errorgraphs.main()
