#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 18:11:12 2015

@author: Satoshi
"""
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

class Data(object):
    u"""舵角、迎角、風速、応力のデータを格納するインスタンスを生成するクラス"""
    def __init__(self,filename, linetype, marker, label):
        self.data = pd.read_csv(filename)
        self.linetype = linetype
        self.marker = marker
        self.label = label
        self.b = 0.404
        self.c = 0.08
        self.S = self.b * self.c
        self.cl = self.c_l()
        self.cd = self.c_d()
        self.cy = self.c_y()
        self.cm = self.c_m()
        self.cr = self.c_r()
        self.cn = self.c_n()
        self.k, self.cdmin, self.clmin = self.polar()
        self.e = (self.k * np.pi * self.b / self.c) ** -1
    
    def c_l(self):
        return self.data["L"]/(0.5*self.data.ix[0,"rho"]*(self.data["wind speed"]**2)*self.S)
        
    def c_d(self):
        return self.data["D"]/(0.5*self.data.ix[0,"rho"]*(self.data["wind speed"]**2)*self.S)

    def c_y(self):
        return self.data["S"]/(0.5*self.data.ix[0,"rho"]*(self.data["wind speed"]**2)*self.S)

    def c_m(self):
        return self.data["B_Pitch"]/(0.5*self.data.ix[0,"rho"]*(self.data["wind speed"]**2)*self.S*self.c)

    def c_r(self):
        return self.data["B_Roll"]/(0.5*self.data.ix[0,"rho"]*(self.data["wind speed"]**2)*self.S*self.b)

    def c_n(self):
        return self.data["B_Yaw"]/(0.5*self.data.ix[0,"rho"]*(self.data["wind speed"]**2)*self.S*self.b)
        
    def fit_func(self,parameter,x,y):
        a = parameter[0]
        b = parameter[1]
        c = parameter[2]
        return x - (a*(y-c)**2+b) # 右辺が求める曲線の形

    def polar(self):
        parameter0=[0.0,0.0,0.0]
        return scipy.optimize.leastsq(self.fit_func,parameter0,args=(self.cd[:8],self.cl[:8]))[0]

    def plot(self, x, y):
        plt.plot(x, y, self.linetype, marker=self.marker, label=self.label, markersize=8, linewidth=1)

    def plot_cl(self):
        self.plot(self.data["AoA"], self.cl)

    def plot_cd(self):
        self.plot(self.data["AoA"], self.cd)

    def plot_cy(self):
        self.plot(self.data["AoA"], self.cy)

    def plot_cm(self):
        self.plot(self.data["AoA"], self.cm)

    def plot_cr(self):
        self.plot(self.data["AoA"], self.cr)

    def plot_cn(self):
        self.plot(self.data["AoA"], self.cn)

    def plot_polar(self, include_e=False):
        if include_e:
            label = self.label + "_e={}".format(round(self.e, 3))
        else:
            label = self.label
        plt.plot(self.cd, self.cl, self.linetype, marker = self.marker, label = label, markersize = 8, linewidth = 1)

    def plot_approximatecurve(self):
        Y = np.arange(-0.6, 1.20, 0.01)
        X = self.k * (Y - self.clmin) ** 2 + self.cdmin
        plt.plot(X, Y, '--', color="k", label="approximate curve e={}".format(round(self.e, 3)))

    def plot_cd_cl2(self):
        self.plot(self.cl**2, self.cd)

    def plot_approximateline(self):
        X = np.arange(0., 1.1, 0.1)
        Y = self.k * (X - self.clmin) ** 2 + self.cdmin
        plt.plot(X**2, Y, '--', color="k", label="approximate line e={}".format(round(self.e, 3)))

    def plot_ld(self):
        self.plot(self.data["AoA"], self.cl/self.cd)


class Offset(Data):
    def c_l(self):
        return self.data["L"]/(0.5*self.data.ix[0,"rho"]*(10**2)*self.S)

    def c_d(self):
        return self.data["D"]/(0.5*self.data.ix[0,"rho"]*(10**2)*self.S)

    def c_y(self):
        return self.data["S"]/(0.5*self.data.ix[0,"rho"]*(10**2)*self.S)

    def c_m(self):
        return self.data["B_Pitch"]/(0.5*self.data.ix[0,"rho"]*(10**2)*self.S*self.c)

    def c_r(self):
        return self.data["B_Roll"]/(0.5*self.data.ix[0,"rho"]*(10**2)*self.S*self.b)

    def c_n(self):
        return self.data["B_Yaw"]/(0.5*self.data.ix[0,"rho"]*(10**2)*self.S*self.b)



class Error(Data):
    def __init__(self, filename, error_filename, linetype, marker, label):
        super(Error, self).__init__(filename, linetype, marker, label)
        self.error = pd.read_csv(error_filename)
        self.errcl = np.absolute(self.cl * self.error["L"])
        self.errcd = np.absolute(self.cd * self.error["D"])
        self.errcy = np.absolute(self.cy * self.error["S"])
        self.errcm = np.absolute(self.cm * self.error["B_Pitch"])
        self.errcr = np.absolute(self.cr * self.error["B_Roll"])
        self.errcn = np.absolute(self.cn * self.error["B_Yaw"])

    def errorbar(self, x, y, err_y, err_x=None):
        plt.errorbar(x, y, err_y, err_x, linestyle="-", label=self.label, linewidth=1.5, elinewidth=2)

    def error_cl(self):
        self.errorbar(self.data["AoA"], self.cl, self.errcl)

    def error_cd(self):
        self.errorbar(self.data["AoA"], self.cd, self.errcd)

    def error_cy(self):
        self.errorbar(self.data["AoA"], self.cy, self.errcy)

    def error_cm(self):
        self.errorbar(self.data["AoA"], self.cm, self.errcm)

    def error_cr(self):
        self.errorbar(self.data["AoA"], self.cr, self.errcr)

    def error_cn(self):
        self.errorbar(self.data["AoA"], self.cn, self.errcn)

    def error_polar(self):
        self.errorbar(self.cd, self.cl, self.errcl, self.errcd)


def result():
    # 読み込むファイルを指定
    inp = pd.read_csv("input.csv")
    results = [Data(row["filename"], row["linetype"], row["marker"], row["label"]) for (index, row) in inp.iterrows()]
    # 図のフォントサイズを指定
    plt.rcParams.update({'font.size':18})

    for result in results:
        result.plot_cl()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.6, 1.4, 0.2))
    plt.ylim(-0.6, 1.2)
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_L$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("CL-alpha")
    plt.close("all")

    for result in results:
        result.plot_cd()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(0.0, 0.35, 0.05))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_D$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("CD-alpha")
    plt.close("all")

    for result in results:
        result.plot_cy()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.02, 0.14, 0.02))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_Y$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("CY-alpha")
    plt.close("all")

    for result in results:
        result.plot_cm()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.3, 0.4, 0.1))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_m$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("Cpitch-alpha")
    plt.close("all")

    for result in results:
        result.plot_cr()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.020, 0.020, 0.002))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_l$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("Croll-alpha")
    plt.close("all")

    for result in results:
        result.plot_cn()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.006, 0.006, 0.002))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_n$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("Cyaw-alpha")
    plt.close("all")

    for result in results:
        result.plot_polar(True)
        # result.plot_approximatecurve()
    plt.xticks(np.arange(0, 0.35, 0.05))
    plt.yticks(np.arange(-0.6, 1.4, 0.2))
    plt.ylim(-0.6, 1.2)
    plt.xlabel("$C_D$", fontsize=24)
    plt.ylabel("$C_L$", fontsize=24)
    plt.grid()
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("polarcurve")
    plt.close("all")

    for result in results:
        result.plot_cd_cl2()
        # result.plot_approximateline()
    plt.xticks(np.arange(-0.2, 1.2, 0.2))
    plt.yticks(np.arange(0., 0.35, 0.05))
    # plt.ylim(-0.6, 1.2)
    plt.xlabel("$C_L^2$", fontsize=24)
    plt.ylabel("$C_D$", fontsize=24)
    plt.grid()
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("cd-cl2")
    plt.close("all")

    for result in results:
        result.plot_ld()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-6, 10, 2))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("L/D", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("ld-alpha")
    plt.close("all")

def result_multi(include_clean=True):
    # 読み込むファイルを指定
    inp = pd.read_csv("input.csv")
    deltas = np.arange(inp.ix[0,"dmin"], inp.ix[0,"dmax"]+inp.ix[0,"dstep"], inp.ix[0,"dstep"], dtype=int)
    clean = Data("result.csv", "k--", "x", "clean")
    for (casenum, delta) in enumerate(deltas):
        results = [Data(row["basefilename"]+"_{}.csv".format(delta), row["linetype"], row["marker"], row["label"])
                   for (index, row) in inp.iterrows()]
        if include_clean:
            results.insert(0, clean)
        # 図のフォントサイズを指定
        plt.rcParams.update({'font.size':18})

        for result in results:
            result.plot_cl()
        plt.xticks(np.arange(-10, 25, 5))
        plt.yticks(np.arange(-0.6, 1.4, 0.2))
        plt.ylim(-0.6, 1.2)
        plt.xlabel("AoA[deg]", fontsize=24)
        plt.ylabel("$C_L$", fontsize=24)
        plt.grid()
        plt.legend(loc="lower right")
        plt.tight_layout(pad=0.05)
        # plt.show()
        plt.savefig("cl-alpha_case{0}_d={1}".format(casenum, delta))
        plt.close("all")

        for result in results:
            result.plot_cd()
        plt.xticks(np.arange(-10, 25, 5))
        plt.yticks(np.arange(0.0, 0.35, 0.05))
        plt.xlabel("AoA[deg]", fontsize=24)
        plt.ylabel("$C_D$", fontsize=24)
        plt.grid()
        plt.legend(loc="upper left")
        plt.tight_layout(pad=0.05)
        # plt.show()
        plt.savefig("cd-alpha_case{0}_d={1}".format(casenum, delta))
        plt.close("all")

        for result in results:
            result.plot_cm()
        plt.xticks(np.arange(-10, 25, 5))
        plt.yticks(np.arange(-0.3, 0.4, 0.1))
        plt.xlabel("AoA[deg]", fontsize=24)
        plt.ylabel("$C_M$", fontsize=24)
        plt.grid()
        plt.legend(loc="lower left")
        plt.tight_layout(pad=0.05)
        # plt.show()
        plt.savefig("cm-alpha_case{0}_d={1}".format(casenum, delta))
        plt.close("all")

        for result in results:
            result.plot_polar(True)
            # result.plot_approximatecurve()
        plt.xticks(np.arange(0, 0.35, 0.05))
        plt.yticks(np.arange(-0.6, 1.4, 0.2))
        plt.ylim(-0.6, 1.2)
        plt.xlabel("$C_D$", fontsize=24)
        plt.ylabel("$C_L$", fontsize=24)
        plt.grid()
        plt.legend(loc="lower right", fontsize=18)
        plt.tight_layout(pad=0.05)
        # plt.show()
        plt.savefig("polarcurve_case{0}_d={1}".format(casenum, delta))
        plt.close("all")

    for result in results:
        result.plot_cd_cl2()
        # result.plot_approximateline()
    plt.xticks(np.arange(-0.2, 1.2, 0.2))
    plt.yticks(np.arange(0., 0.35, 0.05))
    # plt.ylim(-0.6, 1.2)
    plt.xlabel("$C_L^2$", fontsize=24)
    plt.ylabel("$C_D$", fontsize=24)
    plt.grid()
    plt.legend(loc="upper left", fontsize=18)
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("cd-cl2_case{0}_d={1}".format(casenum, delta))
    plt.close("all")

    for result in results:
        result.plot_ld()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-6, 10, 2))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("L/D", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("ld-alpha_case{0}_d={1}".format(casenum, delta))
    plt.close("all")

def offset():
    # 読み込むファイルを指定
    result = Offset("pre_rev.csv", "b-", "x", "pre")
    # 図のフォントサイズを指定
    plt.rcParams.update({'font.size': 18})

    result.plot_cl()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_L$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("offset_CL-alpha")
    plt.close("all")

    result.plot_cd()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_D$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("offset_CD-alpha")
    plt.close("all")

    result.plot_cy()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_Y$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("offset_CY-alpha")
    plt.close("all")

    result.plot_cm()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_m$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("offset_Cpitch-alpha")
    plt.close("all")

    result.plot_cr()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_l$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("offset_Croll-alpha")
    plt.close("all")

    result.plot_cn()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_n$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("offset_Cyaw-alpha")
    plt.close("all")

def rawdata():
    # 読み込むファイルを指定
    result = Data("data_rev.csv", "b-", "x", "data")
    # 図のフォントサイズを指定
    plt.rcParams.update({'font.size': 18})

    result.plot_cl()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$offsetC_L$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("data_cl-alpha")
    plt.close("all")

    result.plot_cd()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_D$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("data_cd-alpha")
    plt.close("all")

    result.plot_cm()
    plt.xticks(np.arange(-10, 25, 5))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_M$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("data_cm-alpha")
    plt.close("all")

def error():
    # 読み込むファイルを指定
    inp = pd.read_csv("input.csv")
    inp = inp.dropna()    ###欠損値あるindexを削除
    results = [Error(row["filename"], row["errorfilename"], row["linetype"], row["marker"], row["label"])
               for (index, row) in inp.iterrows()]
    # 図のフォントサイズを指定
    plt.rcParams.update({'font.size':18})

    for result in results:
        result.error_cl()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.6, 1.4, 0.2))
    plt.ylim(-0.6, 1.2)
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_L$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_CL-alpha")
    plt.close("all")

    for result in results:
        result.error_cd()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(0.0, 0.35, 0.05))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_D$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_CD-alpha")
    plt.close("all")

    for result in results:
        result.error_cy()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.02, 0.14, 0.02))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_Y$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_CY-alpha")
    plt.close("all")

    for result in results:
        result.error_cm()
    plt.xticks(np.arange(-10, 25, 5))
    plt.yticks(np.arange(-0.3, 0.4, 0.1))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_m$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_Cpitch-alpha")
    plt.close("all")

    for result in results:
        result.error_cr()
    plt.xticks(np.arange(-10, 25, 5))
    #plt.yticks(np.arange(0.0, 0.35, 0.05))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_l$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_Croll-alpha")
    plt.close("all")

    for result in results:
        result.error_cn()
    plt.xticks(np.arange(-10, 25, 5))
    #plt.yticks(np.arange(0.0, 0.35, 0.05))
    plt.xlabel("AoA[deg]", fontsize=24)
    plt.ylabel("$C_n$", fontsize=24)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_Cyaw-alpha")
    plt.close("all")


    for result in results:
        result.error_polar()
    plt.xticks(np.arange(0, 0.35, 0.05))
    plt.yticks(np.arange(-0.6, 1.4, 0.2))
    plt.ylim(-0.6, 1.2)
    plt.xlabel("$C_D$", fontsize=24)
    plt.ylabel("$C_L$", fontsize=24)
    plt.grid()
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout(pad=0.05)
    # plt.show()
    plt.savefig("error_polarcurve")
    plt.close("all")