# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT


import os

import numpy as np
import matplotlib.pyplot as plt  # Add this import for plotting

import simpa as sp
from simpa import Tags
from simpa.utils.profiling import profile
from typing import Union

# FIXME temporary workaround for newest Intel architectures
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path_manager = sp.PathManager()
PATH ="/home/sfatima/simpa/simpa_results/2DPhotoacousticYZ_471_800nm.hdf5"


def run_perform_image_reconstruction(SPACING: Union[int, float] = 0.5, path_manager=sp.PathManager(), visualise=True):
    """

    :param SPACING: The simulation spacing between voxels
    :param path_manager: the path manager to be used, typically sp.PathManager
    :param visualise: If VISUALIZE is set to True, the reconstruction result will be plotted
    :return: a run through of the example
    """
    PATH = "/home/sfatima/simpa/simpa_results/2DPhotoacousticYZ_471_800nm.hdf5"
    settings = sp.load_data_field(PATH, Tags.SETTINGS)
    settings[Tags.WAVELENGTH] = settings[Tags.WAVELENGTHS][0]

    settings.set_reconstruction_settings({
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(12e6),
        Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(3e6),
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.SPACING_MM: settings[Tags.SPACING_MM]
    })

    # TODO use the correct device definition here
    device = sp.load_data_field(PATH, Tags.DIGITAL_DEVICE)

    sp.DelayAndSumAdapter(settings).run(device)

    reconstructed_image = sp.load_data_field(PATH, Tags.DATA_FIELD_RECONSTRUCTED_DATA, settings[Tags.WAVELENGTH])
    reconstructed_image = np.squeeze(reconstructed_image)

    if visualise:
        sp.visualise_data(path_to_hdf5_file=PATH,
                          wavelength=settings[Tags.WAVELENGTH],
                          show_reconstructed_data=True,
                          show_xz_only=False,
                          show_initial_pressure=True,
                          show_time_series_data=False,
                          log_scale=False,
                          save_path="2DPAYZ.png")

    # New visualization code
    reconstructed_data = sp.load_data_field(PATH, Tags.DATA_FIELD_RECONSTRUCTED_DATA, settings[Tags.WAVELENGTH])
    if reconstructed_data.ndim == 3:  # Ensure the data has 3 dimensions
        z_pos = int(reconstructed_data.shape[2] / 2)  # Middle z position
        xy_slice = reconstructed_data[:, :, z_pos]    # Extract xy plane

        plt.figure(figsize=(10, 8))
        plt.title("Reconstruction - XY Plane (Circular Cross-Section)")
        plt.imshow(xy_slice, cmap='viridis')
        plt.colorbar()
        plt.savefig("reconstruction_xy_slice.png")
    else:
        print("Reconstructed data does not have the expected dimensions.")

if __name__ == "__main__":
    run_perform_image_reconstruction(SPACING=0.5, path_manager=sp.PathManager(), visualise=True)