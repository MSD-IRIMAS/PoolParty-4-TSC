"""
The following code is inspired by
https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/utils/constants.py
"""

# 128 datasets
UNIVARIATE_DATASET_NAMES_2018 = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                                 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
                                 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                                 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                                 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                                 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                                 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                                 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
                                 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                                 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                                 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                                 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                                 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                                 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                                 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                                 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                                 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                                 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                                 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                                 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                                 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                                 'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                                 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                                 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                                 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                                 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                                 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                                 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                                 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                                 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
                                 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

# 8 datasets
UNIVARIATE_DATASET_NAMES_SHORTLIST = ['ACSF1', 'DiatomSizeReduction', 'Earthquakes', 'OliveOil',
                                      'FaceAll', 'ShapeletSim', 'UMD', 'Wine']
# 32 datasets
UNIVARIATE_DATASET_NAMES_SHORTLIST_M = ['Fish', 'PowerCons', 'Worms', 'WormsTwoClass', 'FacesUCR',
                                        'OSULeaf', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
                                        'Phoneme', 'InsectWingbeatSound', 'Computers', 'WordSynonyms',
                                        'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                                        'SemgHandGenderCh2', 'Yoga', 'Earthquakes', 'EOGHorizontalSignal',
                                        'EOGVerticalSignal', 'LargeKitchenAppliances', 'RefrigerationDevices',
                                        'ScreenType', 'SmallKitchenAppliances', 'MedicalImages', 'Adiac',
                                        'CricketX', 'CricketY', 'CricketZ', 'MiddlePhalanxTW',
                                        'DistalPhalanxOutlineAgeGroup']

# Missing values and unequal length datasets have a "corrected" version located in subfolder of UCR archive.
# Subfolder name is "Missing_value_and_variable_length_datasets_adjusted".
UNIVARIATE_DATASET_NAMES_2018_MISSING_VALUES = ['DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend',
                                                'MelbournePedestrian', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
                                                'AllGestureWiimoteZ', 'GestureMidAirD1', 'GestureMidAirD2',
                                                'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2',
                                                'PickupGestureWiimoteZ', 'PLAID', 'ShakeGestureWiimoteZ']

# 30 datasets (may have unequal length or missing values)
MULTIVARIATE_DATASET_NAMES_2018_ALL = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
                                       'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
                                       'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection',
                                       'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
                                       'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS',
                                       'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
                                       'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump',
                                       'UWaveGestureLibrary']

# 26 datasets with equal length
MULTIVARIATE_DATASET_NAMES_2018 = ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Cricket",
                                   "DuckDuckGeese", "EigenWorms", "Epilepsy", "EthanolConcentration", "ERing",
                                   "FaceDetection", "FingerMovements", "HandMovementDirection", "Handwriting",
                                   "Heartbeat", "Libras", "LSST", "MotorImagery", "NATOPS", "PenDigits", "PEMS-SF",
                                   "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2",
                                   "StandWalkJump", "UWaveGestureLibrary"]
