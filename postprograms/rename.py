#!/usr/bin/env python
# -*- coding: utf-8 -*

from glob import glob
from os import rename

if __name__ == '__main__':
    oldfilename = "result"
    newfilename = "result2"
    oldfiles = glob("./{0}_[0-9].csv".format(oldfilename))