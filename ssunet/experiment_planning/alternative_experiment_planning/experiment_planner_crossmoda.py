import numpy as np
from ssunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from ssunet.paths import *


class ExperimentPlanner3D_v21_crossmoda(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnUNetData_plans_v2.1_crossmoda"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_crossmoda_plans_3D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([1.5, 0.5, 0.5]) # make sure this is float!!!! Not int!


class ExperimentPlanner2D_v21_crossmoda(ExperimentPlanner2D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnUNetData_plans_v2.1_crossmoda"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_crossmoda_plans_3D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([1.5, 0.5, 0.5]) # make sure this is float!!!! Not int!
        