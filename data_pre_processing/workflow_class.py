"""
Stores a workflow class which enables filtering, manipulation, transforming, aggregating and normalising

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""

from config.logger_config import LoggerMixin
from data_pre_processing.transform.transform_class import DataTransformer
from data_pre_processing.manipulator_class import DataManipulator
from data_pre_processing.normalise_class import DataNormaliser
from data_pre_processing.aggregate_class import DataAggregator
from data_pre_processing.export_class import DataExporter
from data_pre_processing.filter.filter_class import DataFilter
from data_pre_processing.file_import_class import DataProcessor


class DataProcessingWorkflow(LoggerMixin):
    """
    Main class for orchestrating the complete data processing workflow
    """
    def __init__(self,
                 debug: bool = False,
                 verbose: bool = True):

        self.processor = DataProcessor(debug=debug,
                                       verbose=verbose)
        self.filter = DataFilter(debug=debug,
                                 verbose=verbose)
        self.aggregator = DataAggregator(debug=debug,
                                        verbose=verbose)
        self.transformer = DataTransformer(debug=debug,
                                           verbose=verbose)
        self.manipulator = DataManipulator(debug=debug,
                                            verbose=verbose)
        self.normalizer = DataNormaliser(debug=debug,
                                            verbose=verbose)

        self.exporter = DataExporter(debug=debug,
                                     verbose=verbose)
