#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 15:41:27 2015

@author: Satoshi
"""
from logging import getLogger, StreamHandler, Formatter, INFO
import output
import graphs


def postprocess_clean():
    # ロガーの設定
    logger = getLogger(__name__)
    shandler = StreamHandler()
    shandler.setFormatter(Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    shandler.setLevel(INFO)
    logger.setLevel(INFO)
    logger.addHandler(shandler)


    logger.info("output_ref")
    output.ref()
    logger.info("output_predata")
    output.pre()
    logger.info("output_data")
    output.data()
    logger.info("ref_momentrevise")
    output.refmomentrevise()
    logger.info("momentrevise")
    output.momentrevise()
    logger.info("output_result")
    output.result()
    logger.info("output_error")
    output.error_drift()
    logger.info("result_graphs")
    graphs.result()
    logger.info("offset_graphs")
    graphs.offset()
    logger.info("rawdata_graphs")
    graphs.rawdata()
    logger.info("error_graphs")
    graphs.error()


def postprocess_fin():
    # ロガーの設定
    logger = getLogger(__name__)
    shandler = StreamHandler()
    shandler.setFormatter(Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    shandler.setLevel(INFO)
    logger.setLevel(INFO)
    logger.addHandler(shandler)

    logger.info("output_ref")
    output.ref()
    logger.info("output_predata")
    output.pre()
    output.data()
    logger.info("output_result")
    output.result(multiplesweep=True)
    # logger.info("output_error")
    # outputerror_drift.main()
    logger.info("result_graphs")
    graphs.result_multi()
    logger.info("offset_graphs")
    graphs.offset()
    # logger.info("rawdata_graphs")
    # rawdatagraphs.main()
    # logger.info("error_graphs")
    # errorgraphs.main()
