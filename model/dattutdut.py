# -*- coding: utf-8 -*-
"""
This is part of the implementation of the DATTUTDUT model.

The DATTUTDUT model estimates evaporative fraction from radiometric 
temperature only. To derive latent and sensible heat estimates an estimate of
net radiation is required.

The DATTUTDUT model was developed by Timmermans, W. J., Kustas, W. P., 
and Andreu, A., 2015.
"""

import numpy as np
import pandas as pd
import numbers
import gdal
import matplotlib.pyplot as plt
import model.dattutdut_utils as utils

sigma = 5.6697*10**(-8)


class BaseModel(object):
    """ Container for the output but also relevant input information of the
    DATTUTDUT model.
    
    Args:
        EF: array of evaporative fraction values (array). dimensionless 
        tmin: (optional) minimum temperature for the computation of EF.
            Originally tmin is calculated as the 0.005 quantile of the 
            lst image.
        tmax: (optional) maximum temperature for the computation of EF.
            Originally tmax is taken as the hottest pixel in the 
            lst image (100 quantile).
        tmin_thres: (optional) quantile threshold to determine tmin 
            (in percent 0 - 100). By default this is 0.005.
        tmax_thres: (optional) quantile threshold to determine tmax
            (in percent 0 - 100). By default this is 100.
        H: (optional) sensible heat flux (array). W m-2
        LE: (optional) latent heat flux (array). W m-2
        g: (optional) soil heat flux (array). W m-2
        Rn: (optional) net radiation (array). W m-2
        
    Returns:
        Object containing all the input information
    """
        
    def set_prop(self, params, value=None):
        """Set model parameters to values passed in params.
        
        Args:
            params: parameter of the DATTUTDUT model. Either a string or dictionary
            values: new value for this parameter. Either a string, number or 
            part of a dictionary
        """
        
        if isinstance(params, dict):
            for param, value in params.items():
                if param in self._param_list:
                    setattr(self, param, value) 
                else:
                    msg = ["'{}' is not a defined model. ".format(param),
                           "Use .get_params to get a list of valid parameters."]
                    raise ValueError("".join(msg)) 
                                     
        elif isinstance(params, (numbers.Number, str)):
            if value:
                if params in self._param_list:
                    setattr(self, params, value)
                else:
                    msg = ["'{}' is not a defined model. ".format(params),
                           "Use .get_params to get a list of valid parameters."]
                    raise ValueError("".join(msg)) 
        else:
            raise ValueError("Format not recognized. Input should be either" + \
                             "a dictionary or a a combination of parameter" + \
                             "name and value.")

    def get_parameter_names(self):
        """Return the list of parameter names."""
        return self._param_list  
    
    
class ModelResults(BaseModel):
    """ Container for the output but also relevant input information of the
    DATTUTDUT model.
    
    Args:
        EF: array of evaporative fraction values (array). dimensionless 
        tmin: (optional) minimum temperature for the computation of EF.
            Originally tmin is calculated as the 0.005 quantile of the 
            lst image.
        tmax: (optional) maximum temperature for the computation of EF.
            Originally tmax is taken as the hottest pixel in the 
            lst image (100 quantile).
        tmin_thres: (optional) quantile threshold to determine tmin 
            (in percent 0 - 100). By default this is 0.005.
        tmax_thres: (optional) quantile threshold to determine tmax
            (in percent 0 - 100). By default this is 100.
        H: (optional) sensible heat flux (array). W m-2
        LE: (optional) latent heat flux (array). W m-2
        G: (optional) soil heat flux (array). W m-2
        Rn: (optional) net radiation (array). W m-2
        
    Returns:
        Object containing all the input information
        
    """
    def __init__(self, EF, tmin, tmax, tmin_thres, tmax_thres, 
                 H=None, LE=None, G=None, Rn=None, albedo=None,
                 config_file=None):
        """ Initialize class and set all variables.
        """
        self.EF = EF
        self.tmin = tmin
        self.tmax = tmax
        self.tmin_thres = tmin_thres
        self.tmax_thres = tmax_thres
        self.H = H
        self.LE = LE
        self.G = G
        self.Rn = Rn
        self.albedo = albedo
        self.input = config_file
        self.input_lst='Undefined'
        self._param_list = ['EF', 'tmin', 'tmax', 'tmin_thres', 'tmax_thres',
                            'H', 'LE', 'g', 'Rn', 'albedo',
                            'input_lst']    
       
class DattutdutModel(BaseModel):
    """ Implementation of the DATTUTDUT Model.
    This model estimates evaporative fraction from radiometric temperature 
    only.
    
    Original publication:
        Timmermans, W. J., Kustas, W. P., and Andreu, A.: Utility of an
        automated thermal-based approach for monitoring evapotranspiration,
        Acta Geophys., 63, 1571–1608, doi:10.1515/acgeo-2015-0016, 2015.
        
    """
    
    def __init__(self):
        """ Initialize DATTUTDUT model. """
        self.tmin_thres = 0.5       # Threshold from the original paper, p.1578 
        self.tmax_thres = 100       # Threshold from the original paper, p.1578
        self.emissivity = 1.0       # Value from the original paper, p.1576
        self.transmissivity = 0.7   # Value from the original paper, p.1576
        self.emis_atm = 1.08*(-np.log(self.transmissivity))**0.265 
        self._param_list = ['tmin_thres', 'tmax_thres', 'emissivity', 
                            'transmissivity', 'emis_atm']
        
    def run_DATTUTDUT(self, config_file, as_dict=True):
        """ Calculate the evaporation fraction (EF) from the lst image. If an
        estimate of shortwave incoming radiation is available it derives 
        net radiation as well as turbulent heat flux estimates from EF.
        
        Args:
            Dictionary with the following information.
            inputLST: array, land surface temperature image (array)
            tmin: (optional) minimum temperature for the computation of EF.
                Originally tmin is calculated as the 0.005 quantile of the 
                lst image.
            tmax: (optional) maximum temperature for the computation of EF.
                Originally tmax is taken as the hottest pixel in the 
                lst image (100 quantile).
            tmin_thres: (optional) quantile threshold to determine tmin 
                (in percent 0 - 100). By default this is 0.005.
            tmax_thres: (optional) quantile threshold to determine tmax
                (in percent 0 - 100). By default this is 100.
            rs: shortwave incoming radiation (either value or array). W m-2.
            g: soil heat flux (either value or array). If a single value is
                given this is interpreted as an net radiation to soil heat 
                flux ratio. If an array of the same size as lst and EF are 
                given this is interpreted as soil heat flux values. W m-2 or
                dimensionless.
            albedo: albedo of the surface (float or array). dimensionless
            ta: air temperature (float). K.
            ea: actual water vapor content in the atmosphere (float). hPa.
            as_dict: (optional) if True the output is returned as dictionary,
                otherwise class object.
        
        Returns:
            An object containing information of modelled fluxes as well as 
            used inputs.
            EF: array of evaporative fraction values (array). dimensionless 
            H: (optional) sensible heat flux (array). W m-2
            LE: (optional) latent heat flux (array). W m-2
            g: (optional) soil heat flux (array). W m-2
            Rn: (optional) net radiation (array). W m-2
            
        """
        
        tmin = config_file['tmin']
        tmax = config_file['tmax']
        tmin_thres = config_file['tmin_thres']
        tmax_thres = config_file['tmax_thres']
        rs = config_file['rs']
        G = config_file['G']
        albedo = config_file['albedo']
        ta = config_file['ta']
        ea = config_file['ea']
        
        
        lst, prj, geo = utils.read_lst_img(config_file['Imagery']['inputLST'], 
                                           na_values=-99.0)
        
        # Check if any ArcGIS NaN values are in the array. Remove all small
        # values.
        lst[lst <= -99] = np.nan
        if np.nanmean(lst) < 200:
            lst += 273.15
            
        if not tmin:
            vals = lst[~np.isnan(lst)]
            if not tmin_thres:
                tmin_thres = self.tmin_thres
            tmin = np.percentile(vals, tmin_thres)
            self.tmin_thres = tmin_thres
        
        if not tmax:
            vals = lst[~np.isnan(lst)]
            if not tmax_thres:
                tmax_thres = self.tmax_thres
            if tmax_thres == 100:
                tmax = np.nanmax(lst)
            else:
                tmax = np.percentile(vals, tmax_thres)
            self.tmax_thres = tmax_thres
       
        # Calculate evaporative fraction        
        EF = self.calc_EF(lst, tmin, tmax)
        EF[EF > 1.0] = 1.0
        EF[EF < 0.0] = 0.0
        # Calculate turbulent heat fluxes
        if rs:
            H, LE, G, Rn, albedo = self.calc_turb_fluxes(lst, EF, rs,
                                                                 tmin, tmax,
                                                                 G, albedo, 
                                                                 ta, ea)
        else:
            H, LE, G, Rn, albedo = (None, None, None, None, None)
        
        
        results = {
                'EF' : EF,
                'H' : H,
                'LE' : LE,
                'G' : G,
                'Rn' : Rn,
                'ancillary' : {
                        'albedo' : albedo,
                        'tmin' : tmin,
                        'tmax' : tmax,
                        'tmin_thres' : tmin_thres,
                        'tmax_thres' : tmax_thres,
                        'lst' : lst}
                }
        
        
        self.write_output_images(results, config_file['Imagery']['outputFile'],
                                 prj, geo)
        
        if not as_dict:
            results = ModelResults(EF, tmin, tmax, tmin_thres, tmax_thres, 
                     H, LE, G, Rn, albedo, config_file)
            
        return results
        
    def calc_EF(self, lst, tmin, tmax):
        """ Calculate the evaporation fraction (EF) from the lst image.
        
        Args:
            lst: array, land surface temperature image (array)
            tmin: (optional) minimum temperature for the computation of EF.
                Originally tmin is calculated as the 0.005 quantile of the 
                lst image.
            tmax: (optional) maximum temperature for the computation of EF.
                Originally tmax is taken as the hottest pixel in the 
                lst image (100 quantile).
            tmin_thres: (optional) quantile threshold to determine tmin 
                (in percent 0 - 100). By default this is 0.005.
            tmax_thres: (optional) quantile threshold to determine tmax
                (in percent 0 - 100). By default this is 100.
                
        Returns:
            An array with evaporative fraction values (EF) for each pixel. 
            
        """
        
        # Calculate evaporative fraction
        EF = (tmax - lst)/(tmax - tmin)   
        
        return EF
        
    def calc_turb_fluxes(self, lst, EF, rs, tmin, tmax,
                         G=None, albedo=None, ta=None, ea=None):
        """Calcualte the turbulent fluxes from an estimate of net radiation, 
        soil heat flux and land surface temperature and calculates evaporative 
        fraction.
        
        Args:
            lst: land surface temperature image (array). K.
            EF: evaporative fraction (array). Dimensionless.
            rs: shortwave incoming radiation (either value or array). W m-2.
            tmin: minimum temperature for the computation of EF.
                Originally tmin is calculated as the 0.005 quantile of the 
                lst image.
            tmax: maximum temperature for the computation of EF.
                Originally tmax is taken as the hottest pixel in the 
                lst image (100 quantile).
            g: (optional) soil heat flux (either value or array). If a single 
                value is given this is interpreted as an net radiation to 
                soil heat flux ratio. 
                If an array of the same size as lst and EF are 
                given this is interpreted as soil heat flux values. W m-2 or
                dimensionless.
            albedo: (optional) albedo of the surface (float or array). 
                Dimensionless
            ta: (optional) air temperature (float). K.
            ea: (optional) actual water vapor content in the atmosphere 
                (float). hPa.
        
        Returns:
            Arrays of sensible, latent, soil heat flux as well as net radiation
            
        """
        
        if not ta:
            # Assumption in the original paper, p.1576
            ta = tmin  
        if ta < 200:
            ta += 273.15
        
        if not albedo:
            # Assumption in the original paper, p.1576
            albedo = 0.05 + (lst - tmin)/(tmax - tmin)*0.2
        
        if ea:
            # Replace the assumed atmospheric emissivty by a better estimate
            # if ea is given
            self.emis_atm = utils.calc_emis_atm(ea, ta)
              
        # Calculate net radiation (Rn)
        Rn = (1 - albedo)*rs+self.emissivity*self.emis_atm*sigma*ta**4 - \
            self.emissivity*sigma*lst**4
        
        # Calculate soil heat flux (g)
        if not isinstance(G, (list, np.ndarray)):
            if not G:
                ratio = 0.05 + (lst - tmin)/(tmax - tmin)*0.4
                G = Rn*ratio
            else:
                G = G*Rn
        elif isinstance(G, list):
            G = np.array(G)
            if not G.shape == lst.shape:
                raise ValueError("Dimensions of soil heat flux and land" + \
                                 "surface temperature do not match.")
        
        # Calculate latent (LE) and sensible (H) 
        LE = EF*(Rn - G)
        H = Rn - G - LE
        
        return H, LE, G, Rn, albedo    

    def write_output_images(self, results, output_dir, prj, geo):
        '''Writes the arrays of an output dictionary which keys match 
            the list in fields to a GeoTIFF. 
            
        Args:
            results: dictionary with the output as well as coresponding input 
                information. 
            output_dir: file name for the output imagery (str).
            prj: projection definition string
            geo: affine transformation coefficients
            
        Returns:
            Writes output geotiff file.
        
        '''
        
        rows,cols=np.shape(results['EF'])
        driver = gdal.GetDriverByName('GTiff')
        if 'LE' in results.keys():
            fields = ['H', 'LE', 'Rn', 'G', 'EF']
        else:
            fields = ['EF']
        nbands=len(fields)
        ds = driver.Create(output_dir, cols, rows, nbands, gdal.GDT_Float32)
        ds.SetGeoTransform(geo)
        ds.SetProjection(prj)
        for i,field in enumerate(fields):
            band=ds.GetRasterBand(i+1)
            band.SetNoDataValue(-99)
            band.WriteArray(results[field])
            band.FlushCache()
        ds.FlushCache()
        del ds
         
         
         