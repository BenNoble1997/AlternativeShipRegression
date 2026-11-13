"""
Purpose of this class is to store functions for building specific filters.
"""
import pandas as pd
import numpy as np
import logging
from config import LoggerMixin
from typing import Tuple


class ShipFilter(LoggerMixin):

    def __init__(self,
                 verbose:bool = True,
                 debug:bool = False,
                 ):
        self.verbose = verbose
        self.debug = debug

        self.set_log_level(verbose=self.verbose,debug=self.debug)

    def length_filter(self,
                      dataset: pd.DataFrame,
                      lbp_col:str='LBP (M)',
                      loa_col:str='LOA (M)',):

        """
        This filter removes ships from your dataset where LBP is greater than LOA
        """
        df =dataset.copy()
        original_len = df.shape[0]

        mask = df[lbp_col] < df[loa_col]

        filtered = df[mask]

        filtered_len = len(filtered)

        self.logger.info(f'length sanity filter applied: {filtered_len}. {original_len-filtered_len} removed')

        return filtered


    def draft_filter(self,
                     dataset: pd.DataFrame,
                     draft_col:str='DRAFT (M)',
                     depth_col:str='DEPTH (M)',):

        """
        This filter removes ships from your dataset where DRAFT is greater than or equal to depth (i.e. your ship is underwater)
        """

        df = dataset.copy()
        original_len = df.shape[0]

        mask = df[depth_col] >= df[draft_col]

        filtered = df[mask]

        filtered_len = len(filtered)

        self.logger.info(f'depth sanity filter applied: {filtered_len}. {original_len - filtered_len} removed')

        return filtered

    def ship_geometric_filter(self,
                              dataset: pd.DataFrame,
                              lbp_col:str='LBP (M)',
                              dwt_col:str="DWT",
                              beam_col:str='BEAM (M)',
                              draft_col:str='DRAFT (M)',
                              rho_sw:float = 1.025):

        """
        This filter removes ships where the DWT/rho*lbp*B*T ratio is greater than 1
        """

        df=dataset.copy()

        original_len = df.shape[0]

        denom = rho_sw*df[lbp_col]*df[beam_col]*df[beam_col]

        geometric_ratio = df[dwt_col]/denom

        geom_mask = geometric_ratio <=1

        filtered = df[geom_mask]

        filtered_len = len(filtered)

        self.logger.info(f'depth sanity filter applied: {filtered_len}. {original_len - filtered_len} removed')

        return filtered



