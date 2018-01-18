# -*- coding: utf-8 -*-
"""
This is part of the implementation of the DATTUTDUT model.

The DATTUTDUT model estimates evaporative fraction from radiometric 
temperature only. To derive latent and sensible heat estimates an estimate of
net radiation is required.

The DATTUTDUT model was developed by Timmermans, W. J., Kustas, W. P., 
and Andreu, A., 2015.
"""


import gdal
import numpy as np

def read_lst_img(img_path, na_values=None):
        """ Read radiometric temperature values as well as georeference data
        from land surface temperature (lst) image.
        
        Args: 
            img_path: path to the lst image that contains the temperature 
                information.
            na_values: float or integer that specifies nan values.
            
        Returns:
            lst: array of lst information
            prj: projection definition string
            geo: affine transformation coefficients
            
        """
        
        fid = gdal.Open(img_path,gdal.GA_ReadOnly)
        lst = fid.GetRasterBand(1).ReadAsArray()
        prj = fid.GetProjection()
        geo = fid.GetGeoTransform()
        
        if lst.dtype == 'uint8':
            lst = lst.astype('float32')
            
        if na_values is not None:
            lst[lst==na_values] = np.nan
            
        return lst, prj, geo

def calc_emis_atm(ea,ta):
    """Estimates the effective atmospheric emissivity for clear sky.

    Args
        ea: float, atmospheric vapour pressure (hPa).
        Ta: float, air temperature (Kelvin).
    
    Returns:
        emis_atm: float, effective atmospheric emissivity (-).

    References:
        Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.
    """
    
    if ta < 200:
        ta += 273.15
    emis_atm=1.24*(ea/ta)**(1./7.)
    return emis_atm     

