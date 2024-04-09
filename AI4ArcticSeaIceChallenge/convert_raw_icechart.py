#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Converts polygon icechart in the raw ASIP3 challenge dataset to SIC, SOD and FLOE charts."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Tore Wulf'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk']
__version__ = '0.0.1'
__date__ = '2022-09-20'

# -- Built-in modules -- #
import copy

# -- Third-party modules -- #
import numpy as np
import xarray as xr

# -- Proprietary modules -- #
from AI4ArcticSeaIceChallenge.utils import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, ICE_STRINGS, GROUP_NAMES, ICECHART_NOT_FILLED_VALUE, ICECHART_UNKNOWN, LOOKUP_NAMES


def convert_polygon_icechart(scene):
    """
    Original polygon_icechart in ASIP3 scenes consists of codes to a lookup table `polygon_codes`.

    This function looks up codes and converts them. 3 variables in the xr scene are created; SIC, SOD and FLOE.
    For SOD and FLOE the partial sea ice concentration is used to determine whether there is a dominant category in a polygon.
    The SOD and FLOE are created using the lookup tables in utils, which dictate the conversion from ice code to class, As multiple codes can be
    converted into a single class, these partial concentrations must also be added. In addition, empty codes, 'not filled values' and unknowns are
    replaced appropriately.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from the ASIP3 challenge dataset.
    """
    # Get codes from polygon_codes.
    codes = np.stack(np.char.split(scene['polygon_codes'].values.astype(str), sep=';'), 0)[SIC_LOOKUP['total_sic_idx']:, :]
    poly_type = np.stack((codes[:, 0] , codes[:, -1]))
    codes = codes[:, :-2].astype(int) 

    # Convert codes to classes for Total and Partial SIC.
    converted_codes = copy.deepcopy(codes)
    for key, value in SIC_LOOKUP.items():
        if type(key) == int:
            for partial_idx in SIC_LOOKUP['sic_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

            tmp = converted_codes[:, SIC_LOOKUP['total_sic_idx']]
            if key in tmp:
                converted_codes[:, SIC_LOOKUP['total_sic_idx']][np.where((tmp == key))[0]] = value

    # Find where partial concentration is empty but total SIC exist.
    ice_ct_ca_empty = np.logical_and(
        converted_codes[:, SIC_LOOKUP['total_sic_idx']] > SIC_LOOKUP[0],
        converted_codes[:, SIC_LOOKUP['sic_partial_idx'][0]] == ICECHART_NOT_FILLED_VALUE)
    # Assign total SIC to partial concentration when empty.
    converted_codes[:, SIC_LOOKUP['sic_partial_idx'][0]][ice_ct_ca_empty] = \
            converted_codes[:, SIC_LOOKUP['total_sic_idx']][ice_ct_ca_empty]

    # Convert codes to classes for partial SOD.
    for key, value in SOD_LOOKUP.items():
        if type(key) == int:
            for partial_idx in SOD_LOOKUP['sod_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

    # Convert codes to classes for partial FLOE.
    for key, value in FLOE_LOOKUP.items():
        if type(key) == int:
            for partial_idx in FLOE_LOOKUP['floe_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

    # Get matching partial ice classes, SOD.
    sod_a_b_bool = converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == \
        converted_codes[:, SOD_LOOKUP['sod_partial_idx'][1]]
    sod_a_c_bool = converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == \
        converted_codes[:, SOD_LOOKUP['sod_partial_idx'][2]]
    sod_b_c_bool = converted_codes[:, SOD_LOOKUP['sod_partial_idx'][1]] == \
        converted_codes[:, SOD_LOOKUP['sod_partial_idx'][2]]

    # Get matching partial ice classes, FLOE.
    floe_a_b_bool = converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == \
        converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][1]]
    floe_a_c_bool = converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == \
        converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][2]]
    floe_b_c_bool = converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][1]] == \
        converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][2]]

    # Remove matches where SOD == -9 and FLOE == -9.
    sod_a_b_bool[np.where(converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == ICECHART_NOT_FILLED_VALUE)] = False
    sod_a_c_bool[np.where(converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == ICECHART_NOT_FILLED_VALUE)] = False
    sod_b_c_bool[np.where(converted_codes[:, SOD_LOOKUP['sod_partial_idx'][1]] == ICECHART_NOT_FILLED_VALUE)] = False
    floe_a_b_bool[np.where(converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == ICECHART_NOT_FILLED_VALUE)] = False
    floe_a_c_bool[np.where(converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == ICECHART_NOT_FILLED_VALUE)] = False
    floe_b_c_bool[np.where(converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][1]] == ICECHART_NOT_FILLED_VALUE)] = False

    # Arrays to loop over to find locations where partial SIC will be combined for SOD and FLOE.
    sod_bool_list = [sod_a_b_bool, sod_a_c_bool, sod_b_c_bool]
    floe_bool_list = [floe_a_b_bool, floe_a_c_bool, floe_b_c_bool]
    compare_indexes = [[0, 1], [0, 2], [1,2]]

    # Arrays to store how much to add to partial SIC.
    sod_partial_add = np.zeros(converted_codes.shape)
    floe_partial_add = np.zeros(converted_codes.shape)

    # Loop to find
    for idx, (compare_idx, sod_bool, floe_bool) in enumerate(zip(compare_indexes, sod_bool_list, floe_bool_list)):
        tmp_sod_bool_indexes = np.where(sod_bool)[0]
        tmp_floe_bool_indexes = np.where(floe_bool)[0]
        if tmp_sod_bool_indexes.size:  #i.e. is array is not empty.
            sod_partial_add[tmp_sod_bool_indexes, SIC_LOOKUP['sic_partial_idx'][compare_idx[0]]] = \
                converted_codes[:, SIC_LOOKUP['sic_partial_idx'][compare_idx[1]]][tmp_sod_bool_indexes]

        if tmp_floe_bool_indexes.size:  # i.e. is array is not empty.
            floe_partial_add[tmp_floe_bool_indexes, SIC_LOOKUP['sic_partial_idx'][compare_idx[0]]] = \
                converted_codes[:, SIC_LOOKUP['sic_partial_idx'][compare_idx[1]]][tmp_floe_bool_indexes]

    # Create arrays for charts.
    scene_tmp = copy.deepcopy(scene['polygon_icechart'].values)
    sic = copy.deepcopy(scene['polygon_icechart'].values)
    sod = copy.deepcopy(scene['polygon_icechart'].values)
    floe = copy.deepcopy(scene['polygon_icechart'].values)

    # Add partial concentrations when classes have been merged in conversion (see SIC, SOD, FLOE tables).
    tmp_sod_added = converted_codes + sod_partial_add.astype(int)
    tmp_floe_added = converted_codes + floe_partial_add.astype(int)

    # Find and replace all codes with SIC, SOD and FLOE.
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(codes.shape[0]):
            code_match = np.where(scene_tmp == converted_codes[i, SIC_LOOKUP['polygon_idx']])
            sic[code_match] = converted_codes[i, SIC_LOOKUP['total_sic_idx']]

            if np.char.lower(poly_type[1, i]) == 'w':
                sic[code_match] = SIC_LOOKUP[0]
            
            # Check if there is a class combined normalized partial concentration, which is dominant in the polygon.
            if np.divide(np.max(tmp_sod_added[i, SIC_LOOKUP['sic_partial_idx']]),
                    tmp_sod_added[i, SIC_LOOKUP['total_sic_idx']]) * 100 >= SOD_LOOKUP['threshold'] * 100:

                # Find dominant partial ice type.
                sod[code_match] = converted_codes[i, SOD_LOOKUP['sod_partial_idx']][
                    np.argmax(tmp_sod_added[i, SIC_LOOKUP['sic_partial_idx']])]
            else:
                sod[code_match] = ICECHART_NOT_FILLED_VALUE
            
            # Check if there is a class combined normalized partial concentration, which is dominant in the polygon.
            if np.divide(np.max(tmp_floe_added[i, SIC_LOOKUP['sic_partial_idx']]),
                    tmp_floe_added[i, SIC_LOOKUP['total_sic_idx']]) * 100 >= FLOE_LOOKUP['threshold'] * 100:
                floe[code_match] = converted_codes[i, FLOE_LOOKUP['floe_partial_idx']][
                    np.argmax(tmp_floe_added[i, SIC_LOOKUP['sic_partial_idx']])]
            else:
                floe[code_match] = ICECHART_NOT_FILLED_VALUE

            if any(converted_codes[i, FLOE_LOOKUP['floe_partial_idx']] == FLOE_LOOKUP['fastice_class']):
                floe[code_match] = FLOE_LOOKUP['fastice_class']

    # Add masked pixels for ambiguous polygons.
    sod[sod == SOD_LOOKUP['invalid']] = SOD_LOOKUP['mask']
    floe[floe == FLOE_LOOKUP['invalid']] = FLOE_LOOKUP['mask']

    # Ensure water is identical across charts.
    sod[sic == SIC_LOOKUP[0]] = SOD_LOOKUP['water']
    floe[sic == SIC_LOOKUP[0]] = FLOE_LOOKUP['water']

    # Add the new charts to scene and add descriptions:
    scene = scene.assign({'SIC': xr.DataArray(sic, dims=scene['polygon_icechart'].dims)})
    scene = scene.assign({'SOD': xr.DataArray(sod, dims=scene['polygon_icechart'].dims)})
    scene = scene.assign({'FLOE': xr.DataArray(floe, dims=scene['polygon_icechart'].dims)})
    
    for chart in CHARTS:
        # Remove any unknowns.
        scene[chart].values[scene[chart].values == ICECHART_UNKNOWN] = LOOKUP_NAMES[chart]['mask']
        
        scene[chart].attrs = ({
            'polygon': ICE_STRINGS[chart],
            'chart_fill_value': LOOKUP_NAMES[chart]['mask']
        })
    
    return scene