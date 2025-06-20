# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
from simpa.utils.profiling import profile
from argparse import ArgumentParser
from typing import Union

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().


@profile
def run_optical_and_acoustic_simulation(spacing: Union[float | int] = 0.2, path_manager=None,
                                        visualise: bool = True):
    """

    :param spacing: The simulation spacing between voxels
    :param path_manager: the path manager to be used, typically sp.PathManager
    :param visualise: If VISUALIZE is set to True, the reconstruction result will be plotted
    :return: a run through of the example
    """
    if path_manager is None:
        path_manager = sp.PathManager()
    VOLUME_TRANSDUCER_DIM_IN_MM = 75
    VOLUME_PLANAR_DIM_IN_MM = 20
    VOLUME_HEIGHT_IN_MM = 25
    RANDOM_SEED = 4711

    # If VISUALIZE is set to True, the simulation result will be plotted
    VISUALIZE = True

    def create_example_tissue():
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_dictionary = sp.Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        tissue_dict = sp.Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=0, thickness_mm=100,
                                                                              molecular_composition=sp.TISSUE_LIBRARY.constant(
                                                                                  0.05, 100, 0.9),
                                                                              priority=1,
                                                                              consider_partial_volume=True,
                                                                              adhere_to_deformation=True)
        tissue_dict["epidermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=1, thickness_mm=0.1,
                                                                                 molecular_composition=sp.TISSUE_LIBRARY.epidermis(),
                                                                                 priority=8,
                                                                                 consider_partial_volume=True,
                                                                                 adhere_to_deformation=True)
        tissue_dict["vessel_1"] = sp.define_circular_tubular_structure_settings(
            tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2 - 10, 0, 5],
            tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2 - 10, VOLUME_PLANAR_DIM_IN_MM, 5],
            molecular_composition=sp.TISSUE_LIBRARY.blood(),
            radius_mm=2, priority=3, consider_partial_volume=True,
            adhere_to_deformation=False
        )
        tissue_dict["vessel_2"] = sp.define_circular_tubular_structure_settings(
            tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2, 0, 10],
            tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM, 10],
            molecular_composition=sp.TISSUE_LIBRARY.blood(),
            radius_mm=3, priority=3, consider_partial_volume=True,
            adhere_to_deformation=False
        )
        return tissue_dict

    # Seed the numpy random configuration prior to creating the global_settings file in
    # order to ensure that the same volume
    # is generated with the same random seed every time.

    np.random.seed(RANDOM_SEED)
    VOLUME_NAME = "CompletePipelineTestMSOT_"+str(RANDOM_SEED)

    general_settings = {
        # These parameters set the general properties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: "CompletePipelineExample_" + str(RANDOM_SEED),
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: spacing,
        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
        Tags.GPU: True,
        Tags.WAVELENGTHS: [700, 800],
        Tags.DO_FILE_COMPRESSION: True,
        Tags.DO_IPASC_EXPORT: True
    }
    settings = sp.Settings(general_settings)
    np.random.seed(RANDOM_SEED)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_example_tissue(),
        Tags.SIMULATE_DEFORMED_LAYERS: True
    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
        Tags.ADDITIONAL_FLAGS: ['--printgpu']  # to print MCX GPU information
    })

    settings.set_acoustic_settings({
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True
    })

    settings.set_reconstruction_settings({
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(8e6),
        Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(0.1e4),
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
        Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
        Tags.DATA_FIELD_DENSITY: 1000,
        Tags.SPACING_MM: spacing
    })

    settings["noise_initial_pressure"] = {
        Tags.NOISE_MEAN: 1,
        Tags.NOISE_STD: 0.01,
        Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
        Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
    }

    settings["noise_time_series"] = {
        Tags.NOISE_STD: 1,
        Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
    }

    # TODO: For the device choice, uncomment the undesired device

    # device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
    #                                                      VOLUME_PLANAR_DIM_IN_MM/2,
    #                                                      0]))
    # device.update_settings_for_use_of_model_based_volume_creator(settings)

    device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                 VOLUME_PLANAR_DIM_IN_MM/2,
                                                                 0]),
                                    field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20]))
    device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                                  pitch_mm=0.25,
                                                                  number_detector_elements=100,
                                                                  field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20])))
    device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))

    SIMULATION_PIPELINE = [
        sp.ModelBasedAdapter(settings),
        sp.MCXAdapter(settings),
        sp.GaussianNoise(settings, "noise_initial_pressure"),
        sp.KWaveAdapter(settings),
        sp.GaussianNoise(settings, "noise_time_series"),
        sp.TimeReversalAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMULATION_PIPELINE, settings, device)

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    if visualise:
        sp.visualise_data(path_to_hdf5_file=settings[Tags.SIMPA_OUTPUT_FILE_PATH],
                          wavelength=WAVELENGTH,
                          show_time_series_data=True,
                          show_initial_pressure=True,
                          show_reconstructed_data=True,
                          log_scale=False,
                          show_xz_only=False,
                          save_path="Photoacoustic.png")


if __name__ == "__main__":
    parser = ArgumentParser(description='Run the optical and acoustic simulation example')
    parser.add_argument("--spacing", default=0.2, type=float, help='the voxel spacing in mm')
    parser.add_argument("--path_manager", default=None, help='the path manager, None uses sp.PathManager')
    parser.add_argument("--visualise", default=True, type=bool, help='whether to visualise the result')
    config = parser.parse_args()

    run_optical_and_acoustic_simulation(spacing=config.spacing, path_manager=config.path_manager,
                                        visualise=config.visualise)
