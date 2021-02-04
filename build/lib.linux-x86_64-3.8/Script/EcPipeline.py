#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pybedtools as bt
from joblib import Parallel, delayed
import pandas as pd
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 10000000)

import multiprocessing
import importlib
import os

from .EcFetch import SoftFetch
from .EcSearch import SearchType
from .EcMerge import MergeReads
from .EcRegion import RegionCat
from .EcFilter import FilterLinks
from .EcCheck import CheckBP
from .EcSeq import SeqEngine
from .EcCNV import  BinBuild, BinCount

class EcDNA():
    'The pipeline used for machine learning models'
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts
        self.arg.scriptdir = os.path.dirname(os.path.realpath(__file__))
        self.arg.datadir   = self.arg.scriptdir + '/../Data/'
        self.arg.Bam    = '%s/%s'%(self.arg.indir, self.arg.bamdir)
        self.arg.Fetch  = '%s/%s'%(self.arg.outdir, self.arg.fetchdir)
        self.arg.Search = '%s/%s'%(self.arg.outdir, self.arg.searchdir)
        self.arg.Merge  = '%s/%s'%(self.arg.outdir, self.arg.mergedir)
        self.arg.Region = '%s/%s'%(self.arg.outdir, self.arg.regiondir)
        self.arg.Cheak  = '%s/%s'%(self.arg.outdir, self.arg.checkdir)
        self.arg.CNV    = '%s/%s'%(self.arg.outdir, self.arg.cnvdir)
        if self.arg.commands == 'Auto':
            self.arg.Pipe = self.arg.pipeline
        else:
            self.arg.Pipe = self.arg.commands

        CORES = multiprocessing.cpu_count()*0.8 if multiprocessing.cpu_count() >8 else 8
        os.environ['NUMEXPR_MAX_THREADS'] = '1000' #str(int(CORES))
        os.environ['PATH'] += ':' + self.arg.bedtools
        importlib.reload(bt)
        bt.set_bedtools_path(self.arg.bedtools)

    def _getinfo(self):
        if os.path.exists(self.arg.infile):
            self.INdf = pd.read_csv(self.arg.infile, sep='\t', comment='#')
        else:
            self.INdf = pd.DataFrame( {'sampleid' : self.arg.infile.split(',')} )
        return self

    def FetchW(self, _l):
        SoftFetch( self.arg, self.log ).GetSoft(_l)

    def SearchW(self, _l):
        SearchType( self.arg, self.log ).TypeBase(_l)

    def MergeW(self, _l):
        MergeReads( self.arg, self.log ).EachEcDNA(_l)

    def RegionW(self, _L):
        RegionCat( self.arg, self.log ).AllEcDNA(_L)

    def FilterW(self, _L):
        FilterLinks( self.arg, self.log ).FormatLink()

    def CheckW(self, _L):
        CheckBP( self.arg, self.log ).BPStat(_L)
        #CheckBP( self.arg, self.log ).PlotLM(_L)
    def SequecW(self, _L):
        SeqEngine( self.arg, self.log ).GetBPSeq(_L)

    def CNVW(self, _L):
        self.arg.splitstr = str( int(self.arg.splitbin/1e3) 
                                    if self.arg.splitbin/1e3 >=1 
                                    else round(self.arg.splitbin/1e3, 1) ) + 'K'
        self.arg.mergestr = str(int(self.arg.mergebin/1e3)) + 'K' \
                                    if self.arg.mergebin/1e6 <1 \
                                    else str(round(self.arg.mergebin/1e6, 1)) + 'M'
        self.arg.binsize = self.arg.mergestr + self.arg.splitstr
        if not self.arg.buildidx:
            self.arg.buildidx = '%s/hg38_split_%s_continue_bin.idx'%(self.arg.datadir, self.arg.binsize)
            self.log.CI('Cannot find the bin file. The defaul file will be used.')
            if not os.path.exists(self.arg.buildidx):
                self.log.CI('Cannot find the defaul bin file. The file will be built.')
                BinBuild( self.arg, self.log ).NoNBed()

        def CNVP(_l):
            BinCount( self.arg, self.log ).DoCoverage(_l)
            BinCount( self.arg, self.log ).CorSeg(_l)
        Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( CNVP  )(_l) for _n, _l in self.INdf.iterrows() )
        #for _n, _l in self.INdf.iterrows():
        #    if _l.sampleid =='BC1':
        #        CNVP(_l)
        #BinCount( self.arg, self.log ).LinePlt(_L)

    def Pipeline(self):
        self._getinfo()
        if 'Fetch' in self.arg.Pipe:
            Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.FetchW )(_l) for _n, _l in self.INdf.iterrows() )
        if 'Search' in self.arg.Pipe:
            Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.SearchW )(_l) for _n, _l in self.INdf.iterrows() )
        if 'Merge' in self.arg.Pipe:
            Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.MergeW  )(_l) for _n, _l in self.INdf.iterrows() )
        if 'Region' in self.arg.Pipe:
            self.RegionW(self.INdf)
        if 'Filter' in self.arg.Pipe:
            self.FilterW(self.INdf)
        if 'Seq' in self.arg.Pipe:
            self.SequecW(self.INdf)
        if 'Check' in self.arg.Pipe:
            self.CheckW(self.INdf)
            #self.PlotLM()
        if 'CNV' in self.arg.Pipe:
            self.CNVW(self.INdf)
