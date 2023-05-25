import copy
import os

import numpy as np
import ntpath

from ikm.model_builder import model_bilder
from ikm.model_builder.time_series_object import TSObject
from ikm.utils.test import Test


def main():
    """
    Implementation of the interpretation algorithm. Computes the errors of an object using leave-one-out cross-validation method.
    """

    number_objs = 0
    classes_ = []
    test = Test()

    objs = [
        [
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.000",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.002",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.021",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.023",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.024",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.025",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.026",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.029",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.032",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.034",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.036",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.043",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.047",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.051",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.065",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.067",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.087",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.030",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.083",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.027",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.033",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.041",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.071",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.073",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.089",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.016",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.017",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.053",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.028",
            #         # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2c0000337.rd.055",

            #

            # "/media/data/lazarenkom98dm/objects-combined/data_157_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_203_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_139_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_263_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_247_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_77_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_17_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_215_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_13_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_237_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_243_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_155_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_27_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_69_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_141_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_241_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_45_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_41_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_57_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_37_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_95_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_229_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_81_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_253_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_105_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_61_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_183_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_89_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_51_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_113_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_175_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_117_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_127_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_195_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_149_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_159_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_7_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_199_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_143_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_21_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_173_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_189_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_75_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_109_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_11_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_3_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_221_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_101_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_121_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_187_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_227_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_9_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_55_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_125_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_137_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_267_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_115_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_239_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_99_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_31_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_153_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_169_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_107_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_87_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_119_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_151_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_251_0.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_245_0.txt",

            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_107_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_117_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_11_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_131_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_137_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_139_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_145_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_151_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_173_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_175_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_181_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_183_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_187_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_197_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_199_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_209_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_221_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_223_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_239_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_249_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_251_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_257_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_25_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_265_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_31_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_3_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_55_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_5_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_79_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_85_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_89_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_95_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_113_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_119_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_159_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_15_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_17_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_1_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_211_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_213_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_267_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_27_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_35_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_59_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_61_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_67_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_81_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_43_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_157_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_243_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_63_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_49_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_241_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_171_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_45_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_41_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_21_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_115_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_215_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_7_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_143_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_263_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_185_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_245_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_227_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_177_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_99_0.txt",

        ],
        [

            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_51_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_9_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_23_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_69_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_165_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_201_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_203_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_123_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_13_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_179_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_255_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_169_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_75_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_149_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_97_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_101_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_103_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_121_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_125_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_135_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_141_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_153_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_189_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_191_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_207_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_229_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_231_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_235_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_259_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_33_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_39_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_47_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_105_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_109_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_111_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_127_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_129_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_133_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_147_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_155_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_161_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_163_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_167_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_193_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_195_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_19_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_205_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_217_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_219_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_225_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_233_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_237_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_247_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_253_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_261_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_29_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_37_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_53_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_57_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_65_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_71_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_73_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_77_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_83_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_87_0.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_91_1.txt",
            "/media/data/lazarenkom98dm/combined-data-left-hemisphere-cz/data_93_1.txt",

            # "/media/data/lazarenkom98dm/objects-combined/data_255_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_123_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_217_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_67_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_145_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_207_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_225_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_165_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_147_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_209_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_65_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_219_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_181_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_29_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_47_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_25_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_197_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_233_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_19_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_161_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_235_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_177_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_131_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_5_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_185_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_135_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_91_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_63_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_167_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_259_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_59_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_201_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_163_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_171_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_39_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_129_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_53_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_73_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_249_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_23_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_49_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_213_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_103_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_211_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_35_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_83_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_97_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_93_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_231_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_205_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_43_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_111_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_133_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_257_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_33_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_191_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_193_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_85_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_15_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_261_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_179_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_71_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_1_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_265_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_223_1.txt",
            # "/media/data/lazarenkom98dm/objects-combined/data_79_1.txt",

            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.014",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.002",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.007",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.009",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.017",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.020",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.022",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.027",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.023",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.024",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.028",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.037",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.061",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.000",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.010",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.012",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.015",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.018",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.019",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.025",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.031",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.039",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.041",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.043",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.045",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.047",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.049",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.055",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.057",
            # "/media/data/lazarenkom98dm/smni_eeg_data_real_preprocessed/co2a0000364.rd.059",

        ]]

    error = 'eucl'
    electrodes = ['Fp1', 'F7', 'F3', 'Cz', 'C3', 'T3', 'P3', 'T5', 'O1']

    class_number = len(objs)
    for i in range(class_number):
        objs_cluster = objs[i]
        number_objs += len(objs_cluster)
        objs_cluster.sort()

        class_ = []
        j = len(objs_cluster) - 1
        while j >= 0:
            data = np.loadtxt(os.path.join(objs_cluster[j]), delimiter=' ')
            filename = ntpath.basename(objs_cluster[j])
            class_.append(TSObject(file_name=filename, data=data, box_cox=True, z_score=True, excl_el=True,
                                   specific_electrodes=electrodes))
            j = j - 1
        classes_.append(class_)

    errors = [0 for c in range(class_number)]
    for cl in range(class_number):
        for obj_num_in_cluster in range(len(classes_[cl])):

            mod_set = copy.deepcopy(classes_)
            test_set = []
            test_set.append(mod_set[cl][obj_num_in_cluster])
            mod_set[cl].pop(obj_num_in_cluster)
            training_mod_set = []
            for i in range(class_number):
                training_mod_set.append(mod_set[i])

            class_models = []
            for i in range(class_number):
                model = model_bilder.ModelBilder.create_model(mod_set[i], error)
                class_models.append(model)

            errors[cl] += test.models_errors(class_models, test_set, cl, error)

    err_number = 0
    print(": err.  = (")
    for error in errors:
        err_number += error
        print(f"{error} ")

    print(f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} %")
    for i in range(len(test.errors_in_models)):
        print(f"{i + 1}:{test.errors_in_models[i]}")


if __name__ == '__main__':
    main()
