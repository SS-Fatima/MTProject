# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.volume_creation_module import VolumeCreationAdapterBase
from simpa.utils.libraries.structure_library import priority_sorted_structures
from simpa.utils import Tags
import numpy as np
from simpa.utils import create_deformation_settings
import torch


class ModelBasedAdapter(VolumeCreationAdapterBase):
    """
    The model-based volume creator uses a set of rules how to generate structures
    to create a simulation volume.
    These structures are added to the dictionary and later combined by the algorithm::

        # Initialise settings dictionaries
        simulation_settings = Settings()
        all_structures = Settings()
        structure = Settings()

        # Definition of en example structure.
        # The concrete structure parameters will change depending on the
        # structure type
        structure[Tags.PRIORITY] = 1
        structure[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        structure[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        all_structures["arbitrary_identifier"] = structure

        simulation_settings[Tags.STRUCTURES] = all_structures

        # ...
        # Define further simulation settings
        # ...

        simulate(simulation_settings)


    """

    def create_simulation_volume(self) -> dict:

        if Tags.SIMULATE_DEFORMED_LAYERS in self.component_settings \
                and self.component_settings[Tags.SIMULATE_DEFORMED_LAYERS]:
            self.logger.debug("Tags.SIMULATE_DEFORMED_LAYERS in self.component_settings is TRUE")
            if Tags.DEFORMED_LAYERS_SETTINGS not in self.component_settings:
                np.random.seed(self.global_settings[Tags.RANDOM_SEED])
                self.component_settings[Tags.DEFORMED_LAYERS_SETTINGS] = create_deformation_settings(
                    bounds_mm=[[0, self.global_settings[Tags.DIM_VOLUME_X_MM]],
                               [0, self.global_settings[Tags.DIM_VOLUME_Y_MM]]],
                    maximum_z_elevation_mm=3,
                    filter_sigma=0,
                    cosine_scaling_factor=1)

        volumes, x_dim_px, y_dim_px, z_dim_px = self.create_empty_volumes()
        global_volume_fractions = torch.zeros((x_dim_px, y_dim_px, z_dim_px),
                                              dtype=torch.float, device=self.torch_device)
        max_added_fractions = torch.zeros((x_dim_px, y_dim_px, z_dim_px), dtype=torch.float, device=self.torch_device)
        wavelength = self.global_settings[Tags.WAVELENGTH]

        for structure in priority_sorted_structures(self.global_settings, self.component_settings):
            self.logger.debug(type(structure))

            structure_properties = structure.properties_for_wavelength(self.global_settings, wavelength)

            structure_volume_fractions = torch.as_tensor(
                structure.geometrical_volume, dtype=torch.float, device=self.torch_device)
            structure_indexes_mask = structure_volume_fractions > 0
            global_volume_fractions_mask = global_volume_fractions < 1
            mask = structure_indexes_mask & global_volume_fractions_mask
            added_volume_fraction = (global_volume_fractions + structure_volume_fractions)

            added_volume_fraction[added_volume_fraction <= 1 & mask] = structure_volume_fractions[
                added_volume_fraction <= 1 & mask]

            selector_more_than_1 = added_volume_fraction > 1
            if torch.any(selector_more_than_1):
                remaining_volume_fraction_to_fill = 1 - global_volume_fractions[selector_more_than_1]
                fraction_to_be_filled = structure_volume_fractions[selector_more_than_1]
                added_volume_fraction[selector_more_than_1] = torch.min(torch.stack((remaining_volume_fraction_to_fill,
                                                                                     fraction_to_be_filled)), 0).values
            for key in volumes.keys():
                if structure_properties[key] is None:
                    continue
                if key == Tags.DATA_FIELD_SEGMENTATION:
                    added_fraction_greater_than_any_added_fraction = added_volume_fraction > max_added_fractions
                    volumes[key][added_fraction_greater_than_any_added_fraction & mask] = structure_properties[key]
                    max_added_fractions[added_fraction_greater_than_any_added_fraction & mask] = \
                        added_volume_fraction[added_fraction_greater_than_any_added_fraction & mask]
                else:
                    if isinstance(structure_properties[key], torch.Tensor):
                        volumes[key][mask] += added_volume_fraction[mask] * \
                            structure_properties[key].to(self.torch_device)[mask]
                    elif isinstance(structure_properties[key], (float, np.float64, int, np.int64)):
                        volumes[key][mask] += added_volume_fraction[mask] * structure_properties[key]
                    else:
                        raise ValueError(f"Unsupported type of structure property. "
                                         f"Was {type(structure_properties[key])}.")

            global_volume_fractions[mask] += added_volume_fraction[mask]

        if (torch.abs(global_volume_fractions[global_volume_fractions > 1]) < 1e-5).any():
            raise AssertionError("Invalid Molecular composition! The volume fractions of all molecules must be"
                                 "exactly 100%!")
        
        # convert volumes back to CPU
        for key in volumes.keys():
            volumes[key] = volumes[key].cpu().numpy().astype(np.float64, copy=False)

        return volumes
