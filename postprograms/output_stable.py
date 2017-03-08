#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

def ref():
    """基準データ（reference）を処理"""
    ref_folder = "reference"
    atmos = pd.read_csv("{}/atmosphere.csv".format(ref_folder))
    atmos.columns = ["rho", "P", "t"]
    caselist = pd.read_csv("{}/case_list.csv".format(ref_folder))

    pre = pd.read_csv("{}/pre.csv".format(ref_folder))
    data = pd.read_csv("{}/data.csv".format(ref_folder))
    post = pd.read_csv("{}/post.csv".format(ref_folder))
    result = data - (pre + post) * 0.5
    result["wind speed"] = data["wind speed"]  # 風速データは通風中のもののみ使用する（オフセットではU=0とする）
    result = pd.concat([caselist.ix[:, 0:4], result.ix[:, 1:], atmos], axis=1)
    result.to_csv("ref.csv", index=False)


def pre():
    """dataをcaselistと統合"""
    atmos = pd.read_csv("atmosphere.csv")
    atmos.columns = ["rho","P","t"]
    caselist = pd.read_csv("case_list_pre.csv")

    pre = pd.read_csv("pre.csv")
    result = pd.concat([caselist,pre.ix[:,1:],atmos],axis=1)
    result.to_csv("pre_.csv", index=False)


def data():
    """dataをcaselistと統合"""
    atmos = pd.read_csv("atmosphere.csv")
    atmos.columns = ["rho","P","t"]
    caselist = pd.read_csv("case_list_pre.csv")

    data = pd.read_csv("data.csv")
    caselist = pd.read_csv("case_list_data.csv")
    result = pd.concat([caselist,data.ix[:,1:],atmos],axis=1)
    result.ix[1,["rho","P","t"]] = result.ix[0,["rho","P","t"]]
    result.to_csv("data_.csv", index=False)

def result(multiplesweep=False):
    """基準データとオフセットで出力を補正し、結果を出力"""
    #データを読み込む（読み込むデータはあらかじめcase_listと統合しておく）
    pre = pd.read_csv("pre_.csv")
    data = pd.read_csv("data_.csv")
    ref = pd.read_csv("ref.csv")
    rho = data.ix[0,"rho"]
    six_force = ["L","D","S","Pitch","Roll","Yaw"] #6分力のラベル
    offset_base = pre.query("flag == 1 & AoA == 0")[six_force].reset_index(drop=True) #基準データの参照値

    #データを測定値(data)とオフセット(drift)に切り分ける
    drift = data.query("flag == 0").reset_index(drop=True)
    data = data.query("flag == 1").reset_index(drop=True)

    #模型にかかる空気力をdriftから引く
    ref_force = ref[six_force]
    wind_speed = drift["wind speed"]**2 / (ref.ix[0,"wind speed"])**2
    wind_force = pd.DataFrame(np.dot(wind_speed[:,np.newaxis],ref_force))
    wind_force.columns = [six_force]
    drift = drift.ix[:,six_force] - wind_force

    #線形補間によりdata測定時のdriftを求め、dataから引く
    n = drift.shape[0]
    mat_offset_base = np.dot(np.ones((n-1,1)),offset_base)
    drift = (drift.head(n-1) + drift.tail(n-1).reset_index(drop=True))/2 - mat_offset_base
    data[six_force] = data[six_force] - drift
    drift.to_csv("drift.csv", index=False)
    offset = np.array(pre.query("flag == 1")[six_force])

    # 1スイープのみの場合
    if not multiplesweep:
        # オフセットを引く
        data_ = data.query("flag == 1")
        data_ = data_.sort_values(by="AoA")
        data_[six_force] = data_[six_force] - offset
        data_["rho"] = np.array([rho] * 12)
        data_.to_csv("result.csv", index=False)

    else:
    # 舵角スイープも行う場合
    # 舵角ごとにオフセットを引く
        for delta in [-10]:
            m = 4+delta/2
            data_ = data.query("flag == 1 & delta_L == {}".format(delta))
            data_ = data_.sort_values(by="AoA")
            data_[six_force] = data_[six_force] - offset[0:m,:]
            data_["rho"] = np.array([rho]*(12+m))
            data_.to_csv("result_{}.csv".format(delta), index=False)

        for delta in np.arange(-8,12,2):
            data_ = data.query("flag == 1 & delta_L == {}".format(delta))
            data_ = data_.sort_values(by="AoA")
            data_[six_force] = data_[six_force] - offset
            data_["rho"] = np.array([rho]*12)
            data_.to_csv("result_{}.csv".format(delta), index=False)

def error_output(six_force, coll_mat, loop_num, data_per_loop):
    """天秤の出力の誤差の絶対値を求める"""
    percision_maker = np.array([0.3675, 0.3675, 0.3675, 0.294, 0.294, 0.294])  # メーカー行列をかけた後の誤差[X,Y,Z,P,Q,R]
    data_per_case = loop_num * data_per_loop
    percision_output = (percision_maker * coll_mat) ** 2
    percision_output = np.sqrt(np.sum(percision_output, axis=1))
    percision_output /= np.sqrt(data_per_case)  # 出力の偶然誤差
    bias_output = six_force * 0.5 * 10 ** -2  # 出力のかたより誤差（おもりの質量の誤差が0.5%として算出）
    err_output = np.sqrt(percision_output ** 2 + bias_output ** 2)
    return err_output


def error_windspeed(windspeed, rho, error_rho, loop_num):
    """風速の相対不確かさを求める"""
    pd = 0.5 * rho * windspeed ** 2
    percision_pd = 0.03 / np.sqrt(loop_num)  # 差圧Pdの偶然誤差
    bias_pd = (0.05 * 10 ** -3) * 999.97 * 9.81 / pd  # 差圧Pdのかたより誤差（マノメータ―の読み取り誤差0.05mmより算出）
    err_windspeed = np.sqrt((0.5 * error_rho) ** 2 + (2 * 0.5 * percision_pd) ** 2 + (0.5 * bias_pd) ** 2)  # Pdの不確かさ幅
    return err_windspeed


def error_rho(filename):
    """空気密度の相対誤差を求める"""
    data = pd.read_csv(filename)
    rho = float(data["rho(Kg/m^3)"])
    p = float(data["Patm(Pa)"])
    t = float(data["Temp(C degree)"])
    delrho_delt = (-(0.348564 * p * (1 - (1.73156 * (10 ** ((7.5 * t) / (t + 237.3)))) / p) / (t + 273.15) ** 2)
                   - (1.38975 * (10 ** ((7.5 * t) / (t + 237.3))) * (
        ((7.5) / (t + 237.3)) - ((7.5 * t) / (t + 237.3) ** 2))) / (t + 273.15))
    delrho_delp = 0.348564 / (t + 273.15)
    delt = 0.025  # 温度計の読み取り誤差
    delp = 0.05 * (1 - 0.000182 * t) * 1013.25 / 760  # 気圧計の読み取り誤差
    err = np.sqrt((delrho_delp * delp) ** 2 + (delrho_delt * delt) ** 2) / rho
    return err


def error_drift():
    """誤差のパーセントを求める"""
    offset_loop_num = 40  # オフセット取得時のループ回数
    rawdata_loop_num = 40  # 通風中のデータ取得時のループ回数
    data_per_loop = 2000  # 1ループ辺りに取得数するデータ点数
    try:
        coll_mat = np.loadtxt("./coll_mat_2016_6_16.csv", delimiter=",")
    except IOError:
        coll_mat = np.loadtxt("C:/Users/kotaro/Documents/Lab/python/postprograms/coll_mat_2016_6_16_new.csv", delimiter=",")

    pre = pd.read_csv("pre_.csv")
    data = pd.read_csv("data_.csv")
    rawdata = data.query("flag == 1").reset_index(drop=True)
    drift = data.query("flag == 0").reset_index(drop=True)
    result = pd.read_csv("result.csv")
    refpre = pd.read_csv("reference/pre.csv")
    refdata = pd.read_csv("reference/data.csv")
    refpost = pd.read_csv("reference/post.csv")

    six_force = ["L", "D", "S", "Pitch", "Roll", "Yaw"]
    pre_six_force = pre.ix[:,six_force].values
    prea0_six_force = pre.query("AoA == 0").ix[:,six_force].values
    driftplus_six_force = drift.ix[1:,six_force].values
    driftminus_six_force = drift.ix[:drift.shape[0]-2, six_force].values
    rawdata_six_force = rawdata.ix[:,six_force].values
    result_six_force = result.ix[:,six_force].values
    refpre_six_force = refpre.ix[:,six_force].values
    refdata_six_force = refdata.ix[:, six_force].values
    refpost_six_force = refpost.ix[:, six_force].values
    result_windspeed = result["wind speed"].values
    rho = float(result.ix[0, "rho"])

    pre_err_out = error_output(pre_six_force, coll_mat, offset_loop_num, data_per_loop)
    prea0_err_out = error_output(prea0_six_force, coll_mat, offset_loop_num, data_per_loop)
    driftplus_err_out = error_output(driftplus_six_force, coll_mat, offset_loop_num, data_per_loop)
    driftminus_err_out = error_output(driftminus_six_force, coll_mat, offset_loop_num, data_per_loop)
    rawdata_err_out = error_output(rawdata_six_force, coll_mat, rawdata_loop_num, data_per_loop)
    refpre_err_out = error_output(refpre_six_force, coll_mat, rawdata_loop_num, data_per_loop)
    refdata_err_out = error_output(refdata_six_force, coll_mat, rawdata_loop_num, data_per_loop)
    refpost_err_out = error_output(refpost_six_force, coll_mat, rawdata_loop_num, data_per_loop)

    err_rho = error_rho("atmosphere.csv")
    result_err_windspeed = error_windspeed(result_windspeed, rho, err_rho, rawdata_loop_num)
    result_err_windspeed = np.dot(result_err_windspeed[:, np.newaxis], np.ones((1,6)))

    result_err_force = np.sqrt(pre_err_out**2 + prea0_err_out**2 + (0.5*driftplus_err_out)**2
                               + (0.5*driftminus_err_out)**2 + rawdata_err_out**2 + (0.5*refpre_err_out)**2
                               + (0.5*refpost_err_out)**2 + refdata_err_out**2) / result_six_force
    result_err = np.sqrt(result_err_force ** 2 +  (2 * result_err_windspeed) ** 2 + err_rho ** 2)
    result_err = pd.DataFrame(result_err, columns=six_force)
    caselist = pd.read_csv("case_list_pre.csv")

    result_err = pd.concat([caselist.ix[:, 0:4], result_err], axis=1)
    result_err.to_csv("error.csv", index=False)


def error_average():
    """誤差のパーセントを求める"""
    offset_loop_num = 40  # オフセット取得時のループ回数
    rawdata_loop_num = 40  # 通風中のデータ取得時のループ回数
    data_per_loop = 2000  # 1ループ辺りに取得数するデータ点数
    try:
        coll_mat = np.loadtxt("./coll_mat_2016_6_16.csv", delimiter=",")
    except IOError:
        coll_mat = np.loadtxt("C:/Users/kotaro/Documents/Lab/python/postprograms/coll_mat_2016_6_16_new.csv", delimiter=",")

    pre = pd.read_csv("pre_.csv")
    data = pd.read_csv("data_.csv")
    post = pd.read_csv("post_.csv")
    result = pd.read_csv("result.csv")
    result_windspeed = result["wind speed"].values

    six_force = ["L", "D", "S", "Pitch", "Roll", "Yaw"]
    pre_six_force = pre.ix[:,six_force].values
    data_six_force = data.ix[:,six_force].values
    post_six_force = post.ix[:,six_force].values
    result_six_force = result.ix[:,six_force].values
    rho = float(result.ix[0, "rho"])

    pre_err_out = error_output(pre_six_force, coll_mat, offset_loop_num, data_per_loop)
    data_err_out = error_output(data_six_force, coll_mat, rawdata_loop_num, data_per_loop)
    post_err_out = error_output(post_six_force, coll_mat, offset_loop_num, data_per_loop)

    err_rho = error_rho("atmosphere.csv")
    result_err_windspeed = error_windspeed(result_windspeed, rho, err_rho, rawdata_loop_num)
    result_err_windspeed = np.dot(result_err_windspeed[:, np.newaxis], np.ones((1,6)))

    result_err_force = np.sqrt((0.5*pre_err_out)**2 + data_err_out**2 + (0.5*post_err_out)**2) / result_six_force
    result_err = np.sqrt(result_err_force ** 2 +  (2 * result_err_windspeed) ** 2 + err_rho ** 2)
    result_err = pd.DataFrame(result_err, columns=six_force)
    caselist = pd.read_csv("case_list_pre.csv")

    result_err = pd.concat([caselist.ix[:, 0:4], result_err], axis=1)
    result_err.to_csv("error.csv", index=False)