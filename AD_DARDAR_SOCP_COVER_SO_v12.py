"""
Originally created by G. Mioche, on 15 oct. 2012
Adapted and revised by C. Bazantay, on 30 apr. 2021
Added day/night mask and surface type mask by A. Dziduch, on 31 aug. 2022 

Before Oct 2017 : DARDAR V1
Between Oct 2017 and Apr 2021 : DARDAR V2
Since Apr 2021 : DARDAR V2.23
Since Aug 2022 : DARDAR V2.3
"""
## IMPORTS ##

from pyhdf.SD import SD,SDC
import h5py
from pyhdf.VS import *
from pyhdf.HDF import *
import pprint

from numpy import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.mlab import griddata

import os
import time
from csv import writer
from time import strptime,strftime
from datetime import datetime
from pickle import STOP

import cProfile
import re

os.chdir("C:/Users/dziduch/Documents\M1-M2")

## DARDAR FONCTION ##

def dardarCloud_geo_projection_v2(p, pfig, filesFolder, lYears, m, d, \
latMin, latMax, resol_lat_deg, longMin, longMax, resol_long_deg, \
segFile=7., altiMin=0., altiMax=15000., \
instruFlag=None, landWaterFlag=None, \
dayNightFlag=None, ScooledOnly_mask=True, ScooledOnlyMPC_mask=True, \
IceAndSupercooled_mask=True, AllScooled_mask=True, IceOnly_mask=True, \
IceContaining_cloud_1_mask=True, IceContaining_cloud_2_mask=True, \
liquidWarm_mask=True, classif_mix_clouds=True, \
iwc_val=False, ext_val=False, re_val=False, lidarRatio_val=False, \
temperature=True, temperature_2m=True, \
CloudAnalysis=True, MeteoConditions=True, Winds=True,\
CloudSatPrecip=True,\
vers="", Surface_Type = True, textFile=True):
    
    """Function analyzing DARDAR-MASK files
    
    Parameters to be set in 'CB_paramFile_GeoProj_COVER_SO_v12'
    ----------
    p : str
        The rootpath
    pfig : str
        The figures folder
    filesFolder : str
        The DARDAR-MASK files folder
    lYears : list
        List of years
    m : list
        List of months
    d : list
        List of days
    latMin, latMax : float, float
        Minimum latitude and maximum latitude of the study, in degrees North
        Defined in 'CB_paramFile_GeoProj_COVER_SO_v10'
    resol_lat_deg : float
        Resolution in latitude of the final mesh grid in output text files
    lonMin, lonMax : float, float
        Minimum latitude and maximum longitude of the study, in degrees East
    resol_long_deg : float
        Resolution in longitude of the final mesh grid in output text files
    segFile : int/float
        Sets the number of days per output text file
    altiMin, altiMax = float, float
        Minimum altitude and maximum altitude of the study, in meters
        Default altiMin = 0
        Default altiMax = 15000
    instruFlag : int
        Used to determine which instruments to take into account
        instruFlag = 1 for lidar only
        instruFlag = 2 for radar only
        instruFlag = 3 for lidar+radar
        Default instruFlag = None
    landWaterFlag : str
        Focus on specific surface state
        landWaterFlag = "water" for water surfaces (including sea ice)
        landWaterFlag = "land" for land surfaces (including inland water)
        Default landWaterFlag = None
    
    
    Parameters for G. Mioche analyses
    ---------------------------------
    ScooledOnly_mask : bool
        Set to True for extraction of supercooled-only pixels
    ScooledOnlyMPC_mask : bool
        Set to True to distinguish supercooled-only pixels that are within a MPC
        or not (two masks)
        Requires
        Ice_Only_mask = True
        IceAndSupercooled_mask = True
    IceAndSupercooled_mask : bool
        Set to True for extraction of ice+supercooled pixels
    AllScooled_mask : bool
        Set to True for extraction of all supercooled-containing pixels
        Returns 2 masks : ice+supercooled and supercooled pixels,
            and ice+supercooled and supercooled in MPC
        Requires ScooledOnlyMPC_mask = True
    IceOnly_mask : bool
        Set to True for extraction of ice-only pixels
    IceContaining_cloud_1_mask : bool
        Set to True for extraction of ice-containing pixels + MPC
    IceContaining_cloud_2_mask : bool
        Set to True for extraction of ice-containing pixels
    liquidWarm_mask : bool
        Set to True for extraction of warm cloud pixels
    classif_mix_clouds : bool
        Set to True to distinguish MPCs
        Defines masks for ice below and/or above supercooled layer
        Requires 
        AllScooled_mask = True
    iwc_val : bool
        Set to True to extract ice water content
    ext_val : bool
        Set to True to extract radar extinction
    re_val : bool
        Set to True to extract effective radius of hydrometeors
    lidarRatio_val : bool
        Set to True to extract lidar ratio
    temperature_2m : bool
        Set to True to unlock 2-meter level air temperature
    
    
    Parameters for C. Bazantay analyses
    -----------------------------------
    CloudAnalysis : bool
        Set to True to unlock cloud classification
        Requires temperature = True
    MeteoConditions : bool
        Set to True for meteorological conditions extraction
    Winds : bool
        Set to True to extract wind fields (only for DARDAR-MASK.v2.23)
    CloudSatPrecip : bool
        Set to True to extract precipitation masks from CloudSat product.
    
    
    Other parameters
    ----------------
    dayNightFlag = str
        Only takes into account observations during day or night
        dayNightFlag = "day" for day-only observations
        Default dayNightFlag = None
    temperature : bool
        Set to True to unlock temperature datasets in DARDAR-MASK files
    vers : str
        Optionnal, can be used to rename output text files
    textFile : bool
        Set to True for output text file creation
        
    
    Returns
    -------
    Text files if textFile = True
    
    """
    
    # Initial prints, summarizing some imput parameters
    print "PARAMETERS:"
    print "year(s):\t\t", lYears
    print "month(s):\t\t", m
    print "day(s):\t\t\t", d
    print "latitude limits:\t", int(latMin), "\tto " , int(latMax), "\t\tdegrees"
    print "longitude limits:\t", int(longMin), "\tto ", int(longMax), "\t\tdegrees"    


    """Initializing the shape of output text files
    
    nbLat, nbLong : int, int
        Number of bins for latitude and longitude
    rangeLat, rangeLong = array, array
        Range of latitude and longitude, based on the chosen resolution
    """
    
    dLat        = resol_lat_deg
    nbLat       = int((latMax-latMin)/resol_lat_deg)         
    rangeLat    = np.array(range(int(latMin)*100,int((latMax+dLat)*100),int(dLat*100)),float32)/100.
    
    dLong       = resol_long_deg
    nbLong      = int((longMax-longMin)/resol_long_deg)       
    rangeLong   = np.array(range(int(longMin)*100,int((longMax+dLat)*100),int(dLong*100)),float32)/100.
    
    # Additionnal prints
    print "latitude resolution and number of bins:",    resol_lat_deg,  nbLat    
    print "longitude resolution and number of bins:",   resol_long_deg, nbLong
    print "=========================================="
    

    """The following loop is for scanning the files folder.
    
    listAllFiles2process = list
        List of all files names, for all years/months/days
    """
    
    listAllFiles2process = [] 
    
    for y in lYears:                                                                  
        listAllDaysFolder = os.listdir(p + y + "/")
        for ndf in listAllDaysFolder:
            if os.path.isdir(p + y + "/" + ndf + "/"):
                """ Detected error (GM, Apr. 2021)
                
                Some files were not put in the correct folder,
                which is breaking the code.
                To avoid this error, it is recommended to add the following line
                to the previous "if" condition :
                
                ("2010_09_23" not in ndf) and ("2011_04_15" not in ndf) and \
                ("2011_03_29" not in ndf) and ("2011_02_11" not in ndf)
                """

                if m != []:
                    for nm in m:
                        if d != []:
                            for nd in d:
                                if "_" + nm + "_" + nd in ndf:
                                    if (os.listdir(p + y + "/" + ndf + "/") != []):
                                        for nf in os.listdir(p + y + "/" + ndf + "/"):
                                            # Extension of DARDAR files BEFORE Version 2.23
                                            # if nf[len(nf)-3:len(nf)] == "hdf":
                                            # Extension of DARDAR files SINCE Version 2.23
                                            if nf[len(nf)-2:len(nf)] == "nc":
                                                listAllFiles2process.append(p + y + "/" + ndf + "/" + nf)
                        else:
                            if "_" + nm + "_" in ndf:
                                if os.listdir(p + y + "/" + ndf + "/") != []: 
                                    for nf in os.listdir(p + y + "/" + ndf + "/"):
                                        # if nf[len(nf)-3:len(nf)] == "hdf":
                                        if nf[len(nf)-2:len(nf)] == "nc":
                                            listAllFiles2process.append(p + y + "/" + ndf + "/" + nf)
                else:
                    for nf in os.listdir(p + y + "/" + ndf + "/"):
                        # if nf[len(nf)-3:len(nf)] == "hdf":
                        if nf[len(nf)-2:len(nf)] == "nc":
                            listAllFiles2process.append(p + y + "/" + ndf + "/" + nf)
            else:
                print ndf, " is not a folder"
    
    # Additional prints
    print "hdf DARDAR_CLOUD files to process."
    print "...PROCESSING..."
    
    
    listFINALE = list(sort(listAllFiles2process)) 
       
    # Initialisation of counters
    comptDay4segmentation = 1.  #Counting the number of processed days
    nfiles = 0.  #Counting the number of files
    
    # Loop to process each file
    for nHDFfiles in range(len(listFINALE)):
        """Different naming for different DARDAR-MASK versions
        
        Before Oct. 2017 :
            julianDayGranule = int(listFINALE[nHDFfiles][-19:-16])
            dayInProcess = strptime(listFINALE[nHDFfiles][-23:-16], "%Y%j")
            YearGranule = (listFINALE[nHDFfiles][-23:-19])
        Between Oct 2017 and Apr 2021 :
            julianDayGranule = int(listFINALE[nHDFfiles][-25:-22])
            dayInProcess = strptime(listFINALE[nHDFfiles][-29:-22], "%Y%j")
            YearGranule = (listFINALE[nHDFfiles][-29:-25])
        """
        
        # Since Apr 2021 with DARDAR Version 2.23 :
        
        julianDayGranule = int(listFINALE[nHDFfiles][-24:-21])
        dayInProcess = strptime(listFINALE[nHDFfiles][-28:-21], "%Y%j")
        YearGranule = (listFINALE[nHDFfiles][-28:-24])  # Not used for CB analyses
        
        nfiles += 1.
        
        # If the day changes, you incremente comptDay4segmentation.
        if nHDFfiles>1 and d==[]:
            """Different naming for different DARDAR-MASK versions
            
            Before Oct. 2017 :
                if julianDayGranule != int(listFINALE[nHDFfiles-1][-19:-16]) and nfiles != 1.:
            Between Oct 2017 and Apr 2021 :
                if julianDayGranule != int(listFINALE[nHDFfiles-1][-25:-22]) and nfiles != 1.:
            """
            # Since Apr 2021 with DARDAR Version 2.23 :
            if julianDayGranule != int(listFINALE[nHDFfiles-1][-24:-21]) and nfiles != 1.:
                comptDay4segmentation += 1.            
            else:
                comptDay4segmentation += 0.
        
        print "processing file:", str(dayInProcess.tm_mday) + "/" + str(dayInProcess.tm_mon)\
        + "/" + str(dayInProcess.tm_year),\
        " file #", nfiles, " - ", nHDFfiles+1, "/", len(listFINALE)
        
        if comptDay4segmentation >= 1. and nfiles == 1.:
            """Different naming for different DARDAR-MASK versions
            
            Before Oct. 2017 :
                day1 = strptime(listFINALE[nHDFfiles][-23:-16], "%Y%j")
            Between Oct 2017 and Apr 2021 :
                day1 = strptime(listFINALE[nHDFfiles][-29:-22], "%Y%j")
            """
            # Since Apr 2021 with DARDAR Version 2.23 :
            day1 = strptime(listFINALE[nHDFfiles][-28:-21], "%Y%j")
        
        """Reading spatial limits of DARDAR-MASK granule
        
        The file is not processed if the observed area is not included in the studied area.
        The following lines work ONLY for DARDAR-MASK.v2.23.
        Datasets need to be closed for hdf4 files with dataset.close() or dataset.endaccess()
        """
        
        hdffile = h5py.File(listFINALE[nHDFfiles], 'r')
        lon1SDS = hdffile['CLOUDSAT_Longitude']
        lon1 = lon1SDS[0]
        lon2SDS = hdffile['CLOUDSAT_Longitude']
        lon2 = lon2SDS[-1]
        
        process = 0
        
        if (lon2 < longMax) and  (lon2 > longMin):
            process += 1
        else:
            process += 0
        
        if (lon1 < longMax) and  (lon1 > longMin):
            process += 1
        else:
            process == 0
        
        # File is processed if observations are included in studied area.
        if process > 0.:
            
            """STEP 1 - Truncate the file for study spatial limits
            
            varSDS : dataset
            var : list, copy of dataset
            """
            
            latSDS = hdffile['CLOUDSAT_Latitude']
            lat = latSDS[()]
            lonSDS = hdffile['CLOUDSAT_Longitude']
            lon = lonSDS[()]
            altiSDS = hdffile['CS_TRACK_Height']
            alti = (altiSDS[()]) * 1000

            idx_latBornes = []
            
            chk = np.copy(lat)
            
            # List of hdf file elements that are included in the studied area.
            idx_latBornes = list(((chk<float(latMax)) & (chk>float(latMin))).nonzero())
            del chk
            
            
            """STEP 2 - Extraction of datasets
            
            To reduce time of calculation, datasets are only saved for precise intervals.
            This avoids to process the entire granule.
            """
            
            if idx_latBornes[0] != []:  # Useless condition because spatial limits were evaluated before.
                # Latitude
                idx1 = idx_latBornes[0][0]  # First hdf file cell that is included in the studied area
                idx2 = idx_latBornes[0][-1]  # Last hdf file cell that is included in the studied area
                
                # Altitude
                # Highest/Lowest altitude, based on the inputs
                idz1 = where(abs(alti-altiMax) == min(abs(alti-altiMax)))[0][0]
                idz2 = where(abs(alti-altiMin) == min(abs(alti-altiMin)))[0][0]
                
                # All the following lists only contain hdf file cells included in previously defined boundaries
                # Latitude, Longitude
                lat = lat[idx1:idx2]
                lon = lon[idx1:idx2]
                # Time
                utcTimeSDS = hdffile['CLOUDSAT_UTC_Time']
                utcTime = utcTimeSDS[idx1:idx2]
                # Altitude
                altiFull = altiSDS[()]
                alti = altiSDS[idz1:idz2]
                # DARDAR categorization
                mask_cloudSDS = hdffile['DARMASK_Simplified_Categorization']
                mask_cloud = mask_cloudSDS[idx1:idx2, idz1:idz2]          
                
                mask_cloud_exting = mask_cloudSDS[idx1:idx2, idz1:idz2] 
                
                # Extraction of thermodynamic variable datasets
                if MeteoConditions == True:
                    
                    # Pressure
                    pressureBulkSDS = hdffile['Pressure']
                    pressureBulk = pressureBulkSDS[idx1:idx2, :]
                    
                    # Specific Humidity
                    SpecificHumidityBulkSDS = hdffile['Specific_humidity']
                    SpecificHumidityBulk = SpecificHumidityBulkSDS[idx1:idx2,:]
                    
                    # SST (only above water)
                    SSTBulkSDS = hdffile['Sea_surface_temperature']
                    SSTBulk = SSTBulkSDS[idx1:idx2]
                    
                    # Surface pressure
                    SurfPressureBulkSDS = hdffile['Surface_pressure']
                    SurfPressureBulk = SurfPressureBulkSDS[idx1:idx2]
                    
                    # Skin temperature != (Surface temperature which corresponds to Temperature 2m)
                    SkinTemperatureBulkSDS = hdffile['Skin_temperature']
                    SkinTemperatureBulk = SkinTemperatureBulkSDS[idx1:idx2]
                
                # Extraction of wind fields
                if Winds == True :
                    
                    # Horizontal wind with eastern direction
                    EastWindBulkSDS = hdffile['U_velocity']
                    EastWindBulk = EastWindBulkSDS[idx1:idx2, :]
                    
                    # Horizontal wind with northern direction
                    NorthWindBulkSDS = hdffile['V_velocity']
                    NorthWindBulk = NorthWindBulkSDS[idx1:idx2, :]
                    
                    # Horizontal surface wind with eastern direction
                    SurfaceEastWindBulkSDS = hdffile['U10_velocity']
                    SurfaceEastWindBulk = SurfaceEastWindBulkSDS[idx1:idx2]
                    
                    # Horizontal surface wind with northern direction
                    SurfaceNorthWindBulkSDS = hdffile['V10_velocity']
                    SurfaceNorthWindBulk = SurfaceNorthWindBulkSDS[idx1:idx2]
                
                if Surface_Type == True : 
                
                    # Extraction of surface type
                    surfaceType_flagSDS = hdffile['CALIOP_IGBP_Surface_Type']
                    surfaceType_flag = surfaceType_flagSDS[idx1:idx2]  
                
                # Extraction of other variables contained in files
                
                # Temperature
                if temperature == True:
                    temperatureBulkSDS = hdffile['Temperature']
                    temperatureBulk = temperatureBulkSDS[idx1:idx2, idz1:idz2]
                    temperatureFull = temperatureBulkSDS[idx1:idx2, :]
                    
                # Surface temperature
                if temperature_2m == True:
                    tempe2mBulkSDS = hdffile['Temperature_2m']
                    tempe2mBulk = tempe2mBulkSDS[idx1:idx2]
                    
                # Ice water content
                if iwc_val == True:
                    iwcBulkSDS = hdffile['iwc']
                    iwcBulk = iwcBulkSDS[idx1:idx2, idz1:idz2]
                    
                # Extinction
                if ext_val == True:
                    extBulkSDS = hdffile['extinction']
                    extBulk = extBulkSDS[idx1:idx2, idz1:idz2]
                    
                # Effective radius
                if re_val == True:
                    reBulkSDS = hdffile['effective_radius']
                    reBulk = reBulkSDS[idx1:idx2, idz1:idz2]
                    
                # Lidar Ratio
                if lidarRatio_val == True:
                    LRBulkSDS = hdffile['lidar_ratio']
                    LRBulk = LRBulkSDS[idx1:idx2, idz1:idz2]
                    
                # Information on instrument
                if instruFlag != None:
                    instru_flagSDS = hdffile['instrument_flag']
                    instru_flag = instru_flagSDS[idx1:idx2, idz1:idz2]

                # Information on surface type
                landWater_flagSDS = hdffile['CALIOP_Land_Water_Mask']
                landWater_flag = landWater_flagSDS[idx1:idx2]      
                
                # Day or night observations
                dayNight_flagSDS = hdffile['CALIOP_Day_Night_Flag']
                dayNight_flag = dayNight_flagSDS[idx1:idx2]
                
                # Close the file for safety
                hdffile.close()
                
                # CloudSat product for precipitations
                # Due to unknown error, if this part is activated, this code needs to be launched month by month.
                if CloudSatPrecip == True :
                    fileNumber = str(listFINALE[nHDFfiles][-28:-9])
                    
                    if dayInProcess.tm_mday/10<1:
                        DayGranule = str('0'+ str(dayInProcess.tm_mday))
                    else :
                        DayGranule = str(dayInProcess.tm_mday)
                    if dayInProcess.tm_mon/10<1:
                        MonthGranule = str('0'+ str(dayInProcess.tm_mon))
                    else :
                        MonthGranule = str(dayInProcess.tm_mon)
                    
                    if YearGranule == "2007" or YearGranule == "2008" or YearGranule == "2009" :
                        CSnameFile = str('Z:/2C-Precip-Column/' + YearGranule + '/'\
                            + YearGranule + '_'+ MonthGranule + '_' + DayGranule + '/'\
                            + fileNumber + '_CS_2C-PRECIP-COLUMN_GRANULE_P1_R05_E02_F00.hdf')
                    else :
                        CSnameFile = str('Z:/2C-Precip-Column/' + YearGranule + '/'\
                            + YearGranule + '_'+ MonthGranule + '_' + DayGranule + '/'\
                            + fileNumber + '_CS_2C-PRECIP-COLUMN_GRANULE_P1_R05_E03_F00.hdf')
                    
                    if os.path.exists(CSnameFile)==True :
                        CSfile = HDF(CSnameFile, SDC.READ)
                        vs = CSfile.vstart()
                        
                        CSprecipSDS = vs.attach('Precip_flag')
                        CSprecip = CSprecipSDS[:]
                        CSprecip = CSprecip[idx1:idx2]
                        
                        vs.end()
                    
                """STEP 3 - Erasing cells of datasets,
                depending on instrument, surface and time conditions
                """
                if instruFlag != None:
                    mask_cloud[(instru_flag!=instruFlag)] = 0.  
                                      
                if landWaterFlag != None:
                    if landWaterFlag == "water":
                        mask_cloud[(landWater_flag!=0) & (landWater_flag!=6) & (landWater_flag!=7)] = 0.                    
                    elif landWaterFlag == "land":
                        mask_cloud[(landWater_flag==0) | (landWater_flag==6) | (landWater_flag==7)] = 0.
                
                # Day/Night mask 
                
                if dayNightFlag != None:
                    if dayNightFlag == "day" :
                        print "DAY ONLY !"
                        mask_cloud[(dayNight_flag!=0)] = 0.
                        
                # Surface type mask
                
                if Surface_Type == True : 
                    surfaceType_flag[surfaceType_flag <= 0] = 0.
                    
                    #Land flag
                    surfaceLand = np.copy(surfaceType_flag) 
                    surfaceLand[(surfaceLand == 1.) | (surfaceLand == 2.) | (surfaceLand == 3.) | (surfaceLand == 4.) | (surfaceLand == 5.) | (surfaceLand == 6.) | (surfaceLand == 7.) | (surfaceLand == 8.) | (surfaceLand ==9.) | (surfaceLand == 10.) | (surfaceLand == 11.) | (surfaceLand == 12.) | (surfaceLand == 13.) | (surfaceLand == 14.) | (surfaceLand == 16.) | (surfaceLand == 18.) | (surfaceLand == 19.) | (surfaceLand == 15.)] = 1. 
                    surfaceLand[(surfaceLand == 17.) | (surfaceLand == 20.)]  = 0.
                    
                    #Water flag 
                    surfaceWater = np.copy(surfaceType_flag)
                    surfaceWater[(surfaceWater != 17.)] = 0.
                    surfaceWater[(surfaceWater == 17.)] = 1.
                    
                    #Ice flag 
                    surfaceIce = np.copy(surfaceType_flag)
                    surfaceIce[(surfaceWater != 20.)] = 0.
                    surfaceIce[(surfaceIce == 20.)] = 1.
                
                """STEP 4 - Mask arrangements
                Several masks are created, depending on what is observed.
                """
                
                # Mask to count the number of relevant observations (clouds, clear sky, aerosols)
                # This excludes bad measurements
                pix_obs = np.copy(mask_cloud)
                pix_obs[(mask_cloud<0.)  | (mask_cloud>=14.)] = 0.
                pix_obs[(mask_cloud>=0.) & (mask_cloud<14.) ] = 1.
                
                
                """ GM analyses.
                The following masks were created by GM, unless noticed ontherwise.
                """
                
                lidar_ext = np.copy(mask_cloud_exting)
                lidar_ext[(lidar_ext!=-2.)] = 0.
                lidar_ext[(lidar_ext==-2.)] = 1.
                
                # Cold clouds only, no rain or liquid warm considered
                pix_in_cloud = np.copy(mask_cloud)
                pix_in_cloud[(pix_in_cloud<=0.) | (pix_in_cloud==6.)\
                | (pix_in_cloud==7.) | (pix_in_cloud==8.) | (pix_in_cloud==11.)\
                | (pix_in_cloud==12.) | (pix_in_cloud==14.) | (pix_in_cloud==15.) ] = 0.
                pix_in_cloud[(pix_in_cloud==1.) | (pix_in_cloud==2.)\
                | (pix_in_cloud==3.) | (pix_in_cloud==4.) | (pix_in_cloud>=5.)\
                | (pix_in_cloud==9.)  | (pix_in_cloud==10.) | (pix_in_cloud==13.) ] = 1.
                pix_in_cloud[(pix_obs==0.)] = 0.
                           
                # Clouds but no precipitation
                Allpix_in_cloud = np.copy(mask_cloud)
                Allpix_in_cloud[(Allpix_in_cloud<=0.) | (Allpix_in_cloud==6.)\
                | (Allpix_in_cloud==8.) | (Allpix_in_cloud==15.)] = 0.
                Allpix_in_cloud[(Allpix_in_cloud==1.) | (Allpix_in_cloud==2.)\
                | (Allpix_in_cloud==3.) | (Allpix_in_cloud==4.) | (Allpix_in_cloud==5.)\
                | (Allpix_in_cloud==7.) | (Allpix_in_cloud==9.) | (Allpix_in_cloud==10.)\
                | (Allpix_in_cloud==11.) | (Allpix_in_cloud==12.) | (Allpix_in_cloud==13.)\
                | (Allpix_in_cloud==14.)] = 1.
                Allpix_in_cloud[(pix_obs==0.)] = 0.
                
                # Mix of ice and supercooled water only
                if IceAndSupercooled_mask == True:
                    mask_cloud_iceAndSupercooled = np.copy(mask_cloud)
                    mask_cloud_iceAndSupercooled[(mask_cloud_iceAndSupercooled!=4.)] = 0.
                    mask_cloud_iceAndSupercooled[(mask_cloud_iceAndSupercooled==4.)] = 1.
                    mask_cloud_iceAndSupercooled[(pix_obs==0.)] = 0.
                    
                # All pixels that contain supercooled water
                if AllScooled_mask == True:
                    mask_cloud_allSupercooled = np.copy(mask_cloud)
                    mask_cloud_allSupercooled[(mask_cloud_allSupercooled!=4.) & (mask_cloud_allSupercooled!=3.)] = 0.
                    mask_cloud_allSupercooled[(mask_cloud_allSupercooled==4.) | (mask_cloud_allSupercooled==3.)] = 1.
                    mask_cloud_allSupercooled[(pix_obs==0.)] = 0.
                    
                # Supercooled-only pixels
                if ScooledOnly_mask == True:
                    mask_cloud_supercooledOnly = np.copy(mask_cloud)
                    mask_cloud_supercooledOnly[(mask_cloud_supercooledOnly!=3.)] = 0.
                    mask_cloud_supercooledOnly[(mask_cloud_supercooledOnly==3.)] = 1.
                    mask_cloud_supercooledOnly[(pix_obs==0.)] = 0.
                
                # Ice-only pixels
                if IceOnly_mask == True:
                    mask_cloud_iceOnly = np.copy(mask_cloud)
                    mask_cloud_iceOnly[(mask_cloud_iceOnly!=1.) & (mask_cloud_iceOnly!=2.)\
                    & (mask_cloud_iceOnly!=9.) & (mask_cloud_iceOnly!=5.) & (mask_cloud_iceOnly!=10.)\
                    & (mask_cloud_iceOnly!=13.)] = 0.
                    mask_cloud_iceOnly[(mask_cloud_iceOnly==1.) | (mask_cloud_iceOnly==2.)\
                    | (mask_cloud_iceOnly==9.) | (mask_cloud_iceOnly==5.) | (mask_cloud_iceOnly==10.)\
                    | (mask_cloud_iceOnly==13.)] = 1.                    
                    mask_cloud_iceOnly[(pix_obs==0.)] = 0.
            
                # Special masks for supercooled water within an MPC of ALONE
                # Requirements detailed in the function definition
                if ScooledOnlyMPC_mask == True:
                    mask_cloud_supercooledOnlyMPC = np.copy(mask_cloud)
                    mask_cloud_supercooledOnlyALONE = np.copy(mask_cloud)

                    mask_cloud_supercooledOnlyMPC[(mask_cloud_supercooledOnlyMPC!=3.)] = 0.
                    mask_cloud_supercooledOnlyMPC[(mask_cloud_supercooledOnlyMPC==3.)] = 1.
                    mask_cloud_supercooledOnlyMPC[(pix_obs==0.)] = 0.
                    
                    mask_cloud_supercooledOnlyALONE[(mask_cloud_supercooledOnlyALONE!=3.)] = 0.
                    mask_cloud_supercooledOnlyALONE[(mask_cloud_supercooledOnlyALONE==3.)] = 1.
                    mask_cloud_supercooledOnlyALONE[(pix_obs==0.)] = 0.
                    
                    for nx in range(len(lat)):  # Scans all latitudes
                        mixTop = -1
                        for nz in range(len(alti)):  # Scans all altitudes                     
                            # If the pixel contains supercooled water only, we look if there is ice below or not.
                            if mask_cloud[nx,nz] == 3.:
                            
                                if mixTop == -1:
                                    mixTop = nz
                                
                                # Liquid layer with ice layer or ice+liquid layer BELOW
                                # 1 layer = 3 pixels                                
                                if nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)]) > 3.\
                                or nansum(mask_cloud_iceAndSupercooled[nx,mixTop:len(alti)]) > 3.:                             
                                    mask_cloud_supercooledOnlyMPC[nx,nz:len(alti)] = mask_cloud_supercooledOnlyMPC[nx,nz:len(alti)] * 1.
                                    mask_cloud_supercooledOnlyALONE[nx,nz:len(alti)] = mask_cloud_supercooledOnlyALONE[nx,nz:len(alti)] * 0.                                    
                                    break
                                
                                # Liquid layer with ice layer or ice+liquid layer BELOW AND ABOVE 
                                elif (nansum(mask_cloud_iceOnly[nx,mixTop-15:mixTop])>3. and nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)])>3.)\
                                    or (nansum(mask_cloud_iceOnly[nx,mixTop-15:mixTop])>3. and nansum(mask_cloud_iceAndSupercooled[nx,mixTop:len(alti)])>3.)\
                                    or (nansum(mask_cloud_iceAndSupercooled[nx,mixTop-15:mixTop])>3. and nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)])>3.)\
                                    or (nansum(mask_cloud_iceAndSupercooled[nx,mixTop-15:mixTop])>3. and nansum(mask_cloud_iceAndSupercooled[nx,mixTop:len(alti)])>3.):
                                    mask_cloud_supercooledOnlyMPC[nx,nz:len(alti)] = mask_cloud_supercooledOnlyMPC[nx,nz:len(alti)] * 1.
                                    mask_cloud_supercooledOnlyALONE[nx,nz:len(alti)] = mask_cloud_supercooledOnlyALONE[nx,nz:len(alti)] * 0.
                                    break
                                    
                                # Liquid layer with ice layer or ice+liquid layer ABOVE
                                if nansum(mask_cloud_iceOnly[nx,mixTop-15:mixTop])>3. or nansum(mask_cloud_iceAndSupercooled[nx,mixTop-15:mixTop])>3.:                                     
                                    mask_cloud_supercooledOnlyMPC[nx,nz:len(alti)] = mask_cloud_supercooledOnlyMPC[nx,nz:len(alti)] * 1.
                                    mask_cloud_supercooledOnlyALONE[nx,nz:len(alti)] = mask_cloud_supercooledOnlyALONE[nx,nz:len(alti)] * 0.
                                    break
                                else:                                                                                                     
                                    mask_cloud_supercooledOnlyMPC[nx,nz] = 0.
                                    mask_cloud_supercooledOnlyALONE[nx,nz] = 1.

                #-------------------------------- GM mask : Special masks Sea2Cloud for ice containing clouds (10/03/2021) --------------------------------#
                #2: ice_only + mix
                #1: ice_only + mix + SC_only_MPC
                
                # All ice-containing pixels, including ice only, mix, ScooledOnlyMPC
                if IceContaining_cloud_1_mask == True:
                    mask_ice_containing_cloud_1 = np.copy(mask_cloud)
                    
                    mask_ice_containing_cloud_1[(mask_ice_containing_cloud_1!=1.)\
                    & (mask_ice_containing_cloud_1!=2.) & (mask_ice_containing_cloud_1!=9.)\
                    & (mask_ice_containing_cloud_1!=5.) & (mask_ice_containing_cloud_1!=10.)
                    & (mask_ice_containing_cloud_1!=13.) & (mask_ice_containing_cloud_1!=4.)
                    &  (mask_cloud_supercooledOnlyMPC!=1.)] = 0.
                    mask_ice_containing_cloud_1[(mask_ice_containing_cloud_1==1.)\
                    | (mask_ice_containing_cloud_1==2.) | (mask_ice_containing_cloud_1==9.)\
                    | (mask_ice_containing_cloud_1==5.) | (mask_ice_containing_cloud_1==10.)\
                    | (mask_ice_containing_cloud_1==13.) | (mask_ice_containing_cloud_1==4.)\
                    | (mask_cloud_supercooledOnlyMPC==1.)] = 1.
                    mask_ice_containing_cloud_1[(pix_obs==0.)] = 0.
                
                # All ice-containing pixels, including ice only, mix
                if IceContaining_cloud_2_mask == True:
                    mask_ice_containing_cloud_2 = np.copy(mask_cloud)
                    
                    mask_ice_containing_cloud_2[(mask_ice_containing_cloud_2!=1.)\
                    & (mask_ice_containing_cloud_2!=2.) & (mask_ice_containing_cloud_2!=9.)\
                    & (mask_ice_containing_cloud_2!=5.) & (mask_ice_containing_cloud_2!=10.)\
                    & (mask_ice_containing_cloud_2!=13.) & (mask_ice_containing_cloud_2!=4.)] = 0.
                    mask_ice_containing_cloud_2[(mask_ice_containing_cloud_2==1.)\
                    | (mask_ice_containing_cloud_2==2.) | (mask_ice_containing_cloud_2==9.)\
                    | (mask_ice_containing_cloud_2==5.) | (mask_ice_containing_cloud_2==10.)\
                    | (mask_ice_containing_cloud_2==13.) | (mask_ice_containing_cloud_2==4.)] = 1.
                    mask_ice_containing_cloud_2[(pix_obs==0.)] = 0.

                # All mixed clouds that contain supercooled water
                if AllScooled_mask == True:
                    mask_cloud_allSupercooled2 = np.copy(mask_cloud)
                    
                    mask_cloud_allSupercooled2[(mask_cloud_iceAndSupercooled!=1.)\
                    & (mask_cloud_supercooledOnlyMPC!=1.)] = 0.
                    mask_cloud_allSupercooled2[(mask_cloud_iceAndSupercooled==1.)\
                    | (mask_cloud_supercooledOnlyMPC==1.)] = 1.
                    mask_cloud_allSupercooled2[(pix_obs==0.)] = 0.
                
                # Warm liquid clouds
                if liquidWarm_mask == True:
                    mask_cloud_liqWarm = np.copy(mask_cloud)
                    
                    mask_cloud_liqWarm[(mask_cloud_liqWarm!=11.)] = 0.
                    mask_cloud_liqWarm[(mask_cloud_liqWarm==11.)] = 1.
                    mask_cloud_liqWarm[(pix_obs==0.)] = 0.
                
                # New MPC masks
                if classif_mix_clouds == True:
                    mask_cloud_sc_ice_BelowAbove = np.copy(mask_cloud_allSupercooled)
                    mask_cloud_sc_ice_BelowOnly  = np.copy(mask_cloud_allSupercooled)
                    mask_cloud_sc_ice_BelowAbove2 = np.copy(mask_cloud_allSupercooled2)
                    mask_cloud_sc_ice_BelowOnly2  = np.copy(mask_cloud_allSupercooled2)
                    
                    for nx in range(len(lat)):
                        mixTop = -1
                        for nz in range(len(alti)):                            
                            if mask_cloud[nx,nz] == 4. or mask_cloud[nx,nz] == 3.:
                            
                                if mixTop == -1:
                                    mixTop = nz
 
                                # Liquid layer with ice layer BELOW ONLY:
                                if nansum(mask_cloud_iceOnly[nx,mixTop-5:mixTop]) <= 3.\
                                and nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)]) > 0.:
                                    mask_cloud_sc_ice_BelowOnly[nx,nz:len(alti)] = mask_cloud_sc_ice_BelowOnly[nx,nz:len(alti)] * 1.
                                    mask_cloud_sc_ice_BelowAbove[nx,nz:len(alti)] = mask_cloud_sc_ice_BelowAbove[nx,nz:len(alti)] * 0.
                                    break
                                    
                                # Liquid layer with ice layer BELOW AND ABOVE:    
                                elif nansum(mask_cloud_iceOnly[nx,mixTop-5:mixTop]) > 3.\
                                and nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)]) > 0.:
                                    mask_cloud_sc_ice_BelowAbove[nx,nz] = 1.
                                    mask_cloud_sc_ice_BelowAbove[nx,nz:len(alti)] = mask_cloud_sc_ice_BelowAbove[nx,nz:len(alti)] * 1.
                                    mask_cloud_sc_ice_BelowOnly[nx,nz:len(alti)] = mask_cloud_sc_ice_BelowOnly[nx,nz:len(alti)] * 0.
                                    break
                                    
                                else:                                                                 
                                    mask_cloud_sc_ice_BelowAbove[nx,nz] = 0.
                                    mask_cloud_sc_ice_BelowOnly[nx,nz] = 0.
                                
                    for nx in range(len(lat)):
                        mixTop = -1
                        for nz in range(len(alti)):
                            # MPC Class (SC+ICE or SC only with ice below or above)
                            if mask_cloud_allSupercooled2[nx,nz] == 1.:
                            
                                if mixTop == -1:
                                    mixTop = nz

                                # Liquid layer with ice layer BELOW ONLY:
                                if nansum(mask_cloud_iceOnly[nx,mixTop-5:mixTop]) <= 3.\
                                and nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)]) > 0.:
                                    mask_cloud_sc_ice_BelowOnly2[nx,nz:len(alti)]  = mask_cloud_sc_ice_BelowOnly2[nx,nz:len(alti)] * 1.
                                    mask_cloud_sc_ice_BelowAbove2[nx,nz:len(alti)] = mask_cloud_sc_ice_BelowAbove2[nx,nz:len(alti)] * 0.
                                    break
                                    
                                # Liquid layer with ice layer BELOW AND ABOVE:    
                                elif nansum(mask_cloud_iceOnly[nx,mixTop-5:mixTop]) > 3.\
                                and nansum(mask_cloud_iceOnly[nx,mixTop:len(alti)]) > 0.:
                                    mask_cloud_sc_ice_BelowAbove2[nx,nz:len(alti)] = mask_cloud_sc_ice_BelowAbove2[nx,nz:len(alti)] * 1.
                                    mask_cloud_sc_ice_BelowOnly2[nx,nz:len(alti)]  = mask_cloud_sc_ice_BelowOnly2[nx,nz:len(alti)] * 0.
                                    break
                                    
                                else:                                                                 
                                    mask_cloud_sc_ice_BelowAbove2[nx,nz] = 0.
                                    mask_cloud_sc_ice_BelowOnly2[nx,nz] = 0.
                    
                    mask_cloud_sc_ice_BelowOnly[(pix_obs==0.)] = 0.
                    mask_cloud_sc_ice_BelowAbove[(pix_obs==0.)] = 0.
                    mask_cloud_sc_ice_BelowOnly2[(pix_obs==0.)] = 0.
                    mask_cloud_sc_ice_BelowAbove2[(pix_obs==0.)] = 0.
                
                """Creation of physical parameters masks
                
                GM comments :
                Only consider physical/technical parameters for cold clouds.
                EXT>0 can happen be it does not interest us here.
                Therefore, it is essential not to take it into account in averages.
                """
                
                # Ice-water content
                if iwc_val == True:
                    iwc = np.copy(iwcBulk)
                    iwc[(iwc<=0.000001)] = np.nan
                    iwc[(pix_in_cloud!=1.)] = np.nan
                    iwc[(pix_obs==0.)] = np.nan
                    
                    if instruFlag != None:
                        iwc[(instru_flag!=instruFlag)] = np.nan
                
                # Temperature
                if temperature == True:
                    tempe = np.copy(temperatureBulk)
                    tempe[(tempe<=0.)] = np.nan
                    # This parameter is used in CB analyses that does take into account all measurements.
                    # tempe[(pix_in_cloud!=1.)] = np.nan
                    tempe[(pix_obs==0.)] = np.nan
                    
                    if instruFlag != None:
                        tempe[(instru_flag!=instruFlag)] = np.nan
                
                # Surface air temperature
                if temperature_2m == True:
                    tempe2m = np.copy(tempe2mBulk)
                    tempe2m[(tempe2m<=0.)] = np.nan
                    # This parameter is used in CB analyses that does take into account all measurements.
                    # tempe[(pix_in_cloud!=1.)] = np.nan
                    # tempe2m[(pix_obs==0.)] = np.nan
                    
                    if instruFlag != None:
                        tempe2m[(instru_flag!=instruFlag)] = np.nan
                
                # Extinction coefficient
                if ext_val == True:
                    ext = np.copy(extBulk)
                    ext[(ext<=0.000001) | (ext>0.1)] = np.nan
                    ext[(pix_in_cloud!=1.)] = np.nan
                    ext[(pix_obs==0.)] = np.nan
                    
                    if instruFlag!=None:
                        ext[(instru_flag!=instruFlag)] = np.nan
                
                # Radius
                if re_val == True:
                    re = np.copy(reBulk)
                    re[(re<=0.000001)] = np.nan
                    re[(pix_in_cloud!=1.)] = np.nan
                    re[(pix_obs==0.)] = np.nan
                    
                    if instruFlag != None:
                        re[(instru_flag!=instruFlag)] = np.nan
                
                # Lidar ratio
                if lidarRatio_val == True:
                    LR = np.copy(LRBulk)
                    LR[(LR<=0.)] = np.nan
                    LR[(pix_in_cloud!=1.)] = np.nan
                    LR[(pix_obs==0.)] = np.nan
                    
                    if instruFlag != None:
                        LR[(instru_flag!=instruFlag)] = np.nan

                """CB analyses
                
                Creation of new masks:
                Thermodynamic variables
                Wind fields
                Clouds
                """

                # Creation of thermodynamic variable masks
                if MeteoConditions == True:
                    pressure = np.copy(pressureBulk)
                    pressure[(pressure<=0.)] = np.nan
                    
                    spec_hum = np.copy(SpecificHumidityBulk)
                    spec_hum[(spec_hum<=0.)] = np.nan
                    
                    rho = np.copy(spec_hum)
                    for i in range (len(lat)):
                        for z in range (len(alti)):
                            rho[i,z] = pressure[i,z] / (287.04 * tempe[i,z])
                            
                    
                    for i in range (len(lat)):
                        for z in range (len(alti)):
                            spec_hum[i,z] = spec_hum[i,z] * rho[i,z] * 1000.
                    
                    SST = np.copy(SSTBulk)
                    SST[(SST<=0.)] = np.nan
                    
                    surf_press = np.copy(SurfPressureBulk)
                    surf_press[(surf_press<=0.)] = np.nan
                    
                    SkinTemp = np.copy(SkinTemperatureBulk)
                    SkinTemp[(SkinTemp<=0.)] = np.nan
                    
                    if instruFlag != None:
                        pressure[(instru_flag!=instruFlag)] = np.nan
                        spec_hum[(instru_flag!=instruFlag)] = np.nan
                        SST[(instru_flag!=instruFlag)] = np.nan
                        surf_press[(instru_flag!=instruFlag)] = np.nan
                
                # Creation of wind field masks
                if Winds == True:
                    Uwind = np.copy(EastWindBulk)
                    Uwind[(Uwind==-999)] = np.nan
                    Vwind = np.copy(NorthWindBulk)
                    Vwind[(Vwind==-999)] = np.nan
                    
                    Surface_Uwind = np.copy(SurfaceEastWindBulk)
                    Surface_Uwind[(Surface_Uwind==-999)] = np.nan
                    Surface_Vwind = np.copy(SurfaceNorthWindBulk)
                    Surface_Vwind[(Surface_Vwind==-999)] = np.nan
                
                if CloudSatPrecip == True :
                    CSRain = np.copy(CSprecip)
                    CSRain[(CSRain!=1) & (CSRain!=2) & (CSRain!=3)] = 0
                    CSRain[(CSRain==1) | (CSRain==2) | (CSRain==3)] = 1
                    
                    CSSnow = np.copy(CSprecip)
                    CSSnow[(CSSnow!=4) & (CSSnow!=5)] = 0
                    CSSnow[(CSSnow==4) | (CSSnow==5)] = 1
                    
                    CSMixed = np.copy(CSprecip)
                    CSMixed[(CSMixed!=6) & (CSMixed!=7)] = 0
                    CSMixed[(CSMixed==6) | (CSMixed==7)] = 1
                    
                    CSAll = np.copy(CSprecip)
                    CSAll[(CSAll==9)] = 0
                    CSAll[(CSAll!=0)] = 1
                    
                """CB cloud-counting method.
                
                The following analysis is different from GM, because the cloud count
                depends on the arrangement of the pixels.
                Details on cloud classification are detailed later.
                
                Parameters/Variables :
                
                Cloud : list
                    Each element is a column of atmosphere.
                    Only the latitude of the point of observation is saved.
                    Elements are equal to the number of cloud observations in each column.
                CloudMatrix : list
                    Each element is a column of atmosphere.
                    Only the latitude of the point of observation is saved.
                    CloudMatrix[i] = 0 if there is no cloud in the column.
                    CloudMatrix[i] = 1 if there are at least 1 cloud in the column.
                
                TopHeightCloud : list
                    Saves the cloudtop altitude of the lowest cloud.
                TopTempCloud : list
                    Saves the cloudtop temperature of the lowest cloud.
                ThicknessCloud : list
                    Saves the thickness of the lowest cloud.
                
                """
                if Surface_Type == True: 
                    land_surface_mask = np.zeros(len(lat))
                    water_surface_mask = np.zeros(len(lat))
                    ice_surface_mask = np.zeros(len(lat))
                    
                    for nx in range(len(lat)):
                        
                        land_surface_mask[nx] = surfaceLand[nx]
                        water_surface_mask[nx] = surfaceWater[nx]
                        ice_surface_mask[nx] = surfaceIce[nx]
                        
                    
                if CloudAnalysis == True:
                    
                    # CLOUDS
                    LidarExtCloud = np.zeros(len(lat))
                    LidarExtCloudMatrix = np.zeros(len(lat))
                    
                    # Mixed phase clouds
                    MPC = np.zeros(len(lat))
                    # Unglaciated supercooled clouds
                    USLC = np.zeros(len(lat))
                    # Unglaciated supercooled clouds (could contain warm pixels)
                    USLC2 = np.zeros(len(lat))
                    # Ice-only clouds
                    IceCloud = np.zeros(len(lat))
                    # Warm-liquid clouds
                    WarmCloud = np.zeros(len(lat))
                    # Cold clouds
                    ColdCloud = np.zeros(len(lat))
                    ColdCloudMatrix = np.zeros(len(lat))
                    # All clouds
                    TotalClouds = np.zeros(len(lat))
                    TotalCloudsMatrix = np.zeros(len(lat))
                    # One cloud per column
                    SingleClouds = np.zeros(len(lat))
                    # Two or more clouds per column
                    MultiClouds = np.zeros(len(lat))
                    # Precipitations
                    Precip = np.zeros(len(lat))
                    PrecipMatrix = np.zeros(len(lat))
                    
                    # Tests
                    ColdRain = np.zeros(len(lat))
                    ColdRainMatrix = np.zeros(len(lat))
                    ColdPrecip = np.zeros(len(lat))
                    ColdPrecipMatrix = np.zeros(len(lat))
                    WarmPrecip = np.zeros(len(lat))
                    WarmPrecipMatrix = np.zeros(len(lat))
                    
                    # Clouds which top is under 273.15K
                    MPCTemp = np.zeros(len(lat))
                    USLCTemp = np.zeros(len(lat))
                    USLC2Temp = np.zeros(len(lat))
                    IceCloudTemp = np.zeros(len(lat))
                    
                    # Properties of clouds
                    MPCMatrix = np.zeros(len(lat))
                    TopHeightMPC = np.zeros(len(lat))
                    TopTempMPC = np.zeros(len(lat))
                    ThicknessMPC = np.zeros(len(lat))
                    
                    USLCMatrix = np.zeros(len(lat))
                    TopHeightUSLC = np.zeros(len(lat))
                    TopTempUSLC = np.zeros(len(lat))
                    ThicknessUSLC = np.zeros(len(lat))
                    
                    USLC2Matrix = np.zeros(len(lat))
                    TopHeightUSLC2 = np.zeros(len(lat))
                    TopTempUSLC2 = np.zeros(len(lat))
                    ThicknessUSLC2 = np.zeros(len(lat))
                    
                    IceCloudMatrix = np.zeros(len(lat))
                    TopHeightIceCloud = np.zeros(len(lat))
                    TopTempIceCloud = np.zeros(len(lat))
                    ThicknessIceCloud = np.zeros(len(lat))
                    
                    WarmCloudMatrix = np.zeros(len(lat))
                    TopHeightWarmCloud = np.zeros(len(lat))
                    TopTempWarmCloud = np.zeros(len(lat))
                    ThicknessWarmCloud = np.zeros(len(lat))
                    
                    # Initialization of lists for thermodynamic variables
                    if MeteoConditions == True:
                        geopotential_700 = np.zeros(len(lat))
                        spec_hum_700 = np.zeros(len(lat))
                        temperature_700 = np.zeros(len(lat))
                        geopotential_850 = np.zeros(len(lat))
                        spec_hum_850 = np.zeros(len(lat))
                        temperature_850 = np.zeros(len(lat))
                        SST_Skin_mask = np.zeros(len(lat))
                        surf_temp = np.zeros(len(lat))
                        surf_press_mask = np.zeros(len(lat))
                        
                        # Tests for integrated specific humidity
                        SH500m = np.zeros(len(lat))
                        SH1000m = np.zeros(len(lat))
                        SH1500m = np.zeros(len(lat))
                        pix500 = np.zeros(len(lat))
                        pix1000 = np.zeros(len(lat))
                        pix1500 = np.zeros(len(lat))
                    
                    # Initialization of lists for wind fields
                    if Winds == True:
                        u_winds_700 = np.zeros(len(lat))
                        v_winds_700 = np.zeros(len(lat))
                        u_winds_850 = np.zeros(len(lat))
                        v_winds_850 = np.zeros(len(lat))
                    
                    # Functions for determining the properties of clouds.
                    def PropertiesMPC(nx,topcloud,chkcloud,remainCLEAR):
                        
                        """The following parameters of this function
                        are valid for all PropertiesXXX functions.
                        
                        Parameters :
                        
                        nx : int
                            Index of latitude
                        topcloud : int
                            Index of altitude, refering to the top of the cloud
                        chkcloud : int
                            Thickness of the cloud, in pixels
                        remainCLEAR : int
                            Sum of clear-sky pixels included in a cloud
                            This value does not take into account the 3-clear-sky-pixel
                            boundary above or below the cloud
                        """
                        
                        MPCMatrix[nx] = 1
                        TopHeightMPC[nx] = alti[topcloud] * 1000  # Convert km to meters
                        TopTempMPC[nx] = tempe[nx,topcloud]
                        ThicknessMPC[nx] = (chkcloud+remainCLEAR) * 60  # Convert pixels to meters
                    
                    def PropertiesUSLC(nx,topcloud,chkcloud,remainCLEAR):
                        USLCMatrix[nx] = 1
                        TopHeightUSLC[nx] = alti[topcloud] * 1000
                        TopTempUSLC[nx] = tempe[nx,topcloud]
                        ThicknessUSLC[nx] = (chkcloud+remainCLEAR) * 60
                    
                    def PropertiesUSLC2(nx,topcloud,chkcloud,remainCLEAR):
                        USLC2Matrix[nx] = 1
                        TopHeightUSLC2[nx] = alti[topcloud] * 1000
                        TopTempUSLC2[nx] = tempe[nx,topcloud]
                        ThicknessUSLC2[nx] = (chkcloud+remainCLEAR) * 60
                    
                    def PropertiesIceCloud(nx,topcloud,chkcloud,remainCLEAR):
                        IceCloudMatrix[nx] = 1
                        TopHeightIceCloud[nx] = alti[topcloud] * 1000
                        TopTempIceCloud[nx] = tempe[nx,topcloud]
                        ThicknessIceCloud[nx] = (chkcloud+remainCLEAR) * 60
                    
                    def PropertiesWarmCloud(nx,topcloud,chkcloud,remainCLEAR):
                        WarmCloudMatrix[nx] = 1
                        TopHeightWarmCloud[nx] = alti[topcloud] * 1000
                        TopTempWarmCloud[nx] = tempe[nx,topcloud]
                        ThicknessWarmCloud[nx] = (chkcloud+remainCLEAR) * 60
                    
                    # Each column is analyzed.
                    for nx in range(len(lat)):
                        # Using GM masks, if the column contains at least 3 cloudy pixels,
                        # there is potentially a cloud in the column.
                        
                        if nansum(Allpix_in_cloud[nx,:]) >= 3:
                            # Initialisation of pixel counters
                            chkSCWice = 0  # Mix
                            chkICE = 0  # Ice-only
                            chkWARM = 0  # Liquid warm
                            chkSCW = 0  # Supercooled
                            chkCLEAR = 0  # Clear sky / Aerosols...
                            remainCLEAR = 0  # Number of isolate clear-sky pixels that are between cloudy pixels.
                            chkcloud = 0  # Sum of pixels contained in a cloud
                            topcloud = 0  # Index of altitude for the cloudtop
                            chkmeteo = 0
                            chkprecip = 0  # Precipitations
                            chkLIDAR = 0
                            
                            # Tests
                            chkcoldR = 0
                            chkColdPrecip = 0
                            chkWarmPrecip = 0
                            
                            # Run through all altitude, starting from the bottom.
                            for nz in range(len(alti)-1,-1,-1):
                                
                                # SCW+ice pixels
                                if mask_cloud[nx,nz] == 4:
                                    chkcloud += 1
                                    chkSCWice += 1
                                    topcloud = nz
                                    
                                    """Clear-sky pixels issues :
                                    
                                    If a cloudy pixel is detected after a clear-sky pixel,
                                    the cloud is considered unfinished in space.
                                    Therefore, the cloud is thicker by 2 pixels (the previous clear-sky pixel
                                    and the cloudy pixel that is detected now).
                                    As this is not a clear-sky pixel, the number of boundary clear-sky pixels = 0.
                                    """
                                    if chkCLEAR != 0 :
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                
                                # Ice pixels
                                if mask_cloud[nx,nz] == 1 or mask_cloud[nx,nz] == 2\
                                or mask_cloud[nx,nz] == 9 or mask_cloud[nx,nz] == 10:
                                    chkcloud += 1
                                    chkICE += 1
                                    topcloud = nz
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                
                                # Warm pixels
                                if mask_cloud[nx,nz] == 7 or mask_cloud[nx,nz] == 11\
                                or mask_cloud[nx,nz] == 12 or mask_cloud[nx,nz] == 14\
                                or mask_cloud[nx,nz] == 5 or mask_cloud[nx,nz] == 13:
                                    chkcloud += 1
                                    chkWARM += 1
                                    topcloud = nz
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                
                                # SCW only pixels
                                if mask_cloud[nx,nz] == 3:
                                    chkcloud += 1
                                    chkSCW += 1
                                    topcloud = nz
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                
                                # Precipitations
                                if mask_cloud[nx,nz] == 5 or mask_cloud[nx,nz] == 7\
                                or mask_cloud[nx,nz] == 12 or mask_cloud[nx,nz] == 13 or mask_cloud[nx,nz] == 14:
                                    chkprecip += 1
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                
                                # Cold rain (TEST)
                                if mask_cloud[nx,nz] == 5 :
                                    chkcoldR += 1
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                
                                # Warm Precip (TEST)
                                if mask_cloud[nx,nz] == 7 or mask_cloud[nx,nz] == 12 or mask_cloud[nx,nz] == 14:
                                    chkWarmPrecip += 1
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                        
                                # Cold Precip (TEST)
                                if (mask_cloud[nx,len(alti)-1]==1 or mask_cloud[nx,len(alti)-1]==2\
                                or mask_cloud[nx,len(alti)-1]==9 or mask_cloud[nx,len(alti)-1]==10)\
                                and (mask_cloud[nx,len(alti)-2]==1 or mask_cloud[nx,len(alti)-2]==2\
                                or mask_cloud[nx,len(alti)-2]==9 or mask_cloud[nx,len(alti)-2]==10)\
                                and (mask_cloud[nx,len(alti)-3]==1 or mask_cloud[nx,len(alti)-3]==2\
                                or mask_cloud[nx,len(alti)-3]==9 or mask_cloud[nx,len(alti)-3]==10):
                                    chkColdPrecip += 1
                                    
                                    if chkCLEAR != 0:
                                        remainCLEAR += chkCLEAR
                                        chkCLEAR = 0
                                else :
                                    if mask_cloud[nx,nz] == 5 :
                                        chkColdPrecip += 1
                                        
                                        if chkCLEAR != 0:
                                            remainCLEAR += chkCLEAR
                                            chkCLEAR = 0
                                    
                                # Clear sky pixels
                                if (mask_cloud[nx,nz]==-2 or mask_cloud[nx,nz]==-1 or mask_cloud[nx,nz]==0\
                                or mask_cloud[nx,nz]==6 or mask_cloud[nx,nz]==8 or mask_cloud[nx,nz]==15)\
                                and chkcloud > 0:
                                    chkCLEAR += 1
                                    if lidar_ext[nx, nz] == 1 : 
                                        chkLIDAR +=1
                                    
                                
                                #il faut que je regarde si c'est -2 avant le début d'un nuage
                                
                                # CASE 1 : 3 consecutive clear-sky pixels were detected.
                                if chkCLEAR == 3:
                                    # 3 consecutive pixels = 1 cloud                     
                                    if chkcloud >= 3:
                                        TotalClouds[nx] += 1
                                        if TotalClouds[nx] == 1:
                                            TotalCloudsMatrix[nx] = 1
                                        if chkLIDAR >= 1: 
                                            LidarExtCloud[nx] +=1 
                                            LidarExtCloudMatrix[nx] = 1
                                        # if the cloud contains at least 1 cold pixel, it is a cold cloud
                                        if (chkSCWice+chkICE+chkSCW) >= 1:
                                            ColdCloud[nx] += 1
                                            if ColdCloud[nx] == 1:
                                                ColdCloudMatrix[nx] = 1
                                            # MPC
                                            if chkSCWice >= 1:
                                                MPC[nx] += 1
                                                if MPC[nx] == 1:
                                                    PropertiesMPC(nx,topcloud,chkcloud,remainCLEAR)
                                                # MPC with cloudtop temperature below 0°C (same for all XXXTemp masks).
                                                if tempe[nx,topcloud] <= 273.15:
                                                    MPCTemp[nx] += 1
                                            # MPC
                                            elif (chkSCWice==0 and chkICE>=1 and chkWARM>=1)\
                                            or (chkSCWice==0 and chkICE>=1 and chkSCW>=1):
                                                MPC[nx] += 1
                                                if MPC[nx] == 1:
                                                    PropertiesMPC(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    MPCTemp[nx] += 1
                                            # Ice-only cloud
                                            elif chkICE>=1 and chkSCWice==0 and chkWARM==0 and chkSCW==0 :
                                                IceCloud[nx] += 1
                                                if IceCloud[nx] == 1:
                                                    PropertiesIceCloud(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    IceCloudTemp[nx] += 1
                                            # USLC
                                            elif chkSCW>=1 and chkICE==0 and chkSCWice==0 :
                                                #USLC without warm
                                                if chkWARM == 0 :
                                                    USLC[nx] += 1
                                                    if USLC[nx] == 1:
                                                        PropertiesUSLC(nx,topcloud,chkcloud,remainCLEAR)
                                                    if tempe[nx,topcloud] <= 273.15:
                                                        USLCTemp[nx] += 1
                                                #USLC with warm
                                                if chkWARM>=0 :
                                                    USLC2[nx] += 1
                                                    if USLC2[nx] == 1:
                                                        PropertiesUSLC2(nx,topcloud,chkcloud,remainCLEAR)
                                                    if tempe[nx,topcloud] <= 273.15:
                                                        USLC2Temp[nx] += 1
                                        # The cloud is warm if it does not contain any cold pixel.
                                        elif chkWARM>=1 and chkICE==0 and chkSCWice==0 and chkSCW==0 :
                                            WarmCloud[nx] += 1
                                            if WarmCloud[nx] == 1:
                                                PropertiesWarmCloud(nx,topcloud,chkcloud,remainCLEAR)
                                    
                                    # 3 consecutive precipitation pixels = 1 precipitation layer
                                    # Cumulative with clouds
                                    if chkprecip >= 3:
                                        Precip[nx] += 1
                                        if Precip[nx] == 1:
                                            PrecipMatrix[nx] = 1
                                    
                                    # Tests        
                                    if chkcoldR >= 3:
                                        ColdRain[nx] += 1
                                        if ColdRain[nx] == 1:
                                            ColdRainMatrix[nx] = 1
                                    if chkColdPrecip >= 3:
                                        ColdPrecip[nx] += 1
                                        if ColdPrecip[nx] == 1:
                                            ColdPrecipMatrix[nx] = 1
                                    if chkWarmPrecip >= 3:
                                        WarmPrecip[nx] += 1
                                        if WarmPrecip[nx] == 1:
                                            WarmPrecipMatrix[nx] = 1
                                        
                                    # Re-initialisation of counters
                                    chkSCWice = 0
                                    chkICE = 0
                                    chkWARM = 0
                                    chkSCW = 0
                                    chkCLEAR = 0
                                    remainCLEAR = 0
                                    chkLIDAR = 0
                                    chkcloud = 0
                                    topcloud = 0
                                    chkprecip = 0
                                    chkcoldR = 0
                                    chkColdPrecip = 0
                                    chkWarmPrecip = 0
                                
                                # CASE 2 : The maximum altitude is reached
                                elif nz == 0 :
                                    # There are at least 3 consecutive cloudy pixels.
                                    # This case is not conditioned by the 3-clear-sky-pixel boundaries.
                                    if chkcloud >= 3:
                                        TotalClouds[nx] += 1
                                        if TotalClouds[nx] == 1:
                                            TotalCloudsMatrix[nx] = 1
                                        
                                        if (chkSCWice+chkICE+chkSCW) >= 1:
                                            ColdCloud[nx] += 1
                                            if ColdCloud[nx] == 1:
                                                ColdCloudMatrix[nx] = 1
                                            # MPC
                                            if chkSCWice>=1 :
                                                MPC[nx] += 1
                                                if MPC[nx] == 1:
                                                    PropertiesMPC(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    MPCTemp[nx] += 1
                                            # MPC
                                            elif (chkSCWice==0 and chkICE>=1 and chkWARM>=1)\
                                            or (chkSCWice==0 and chkICE>=1 and chkSCW>=1) :
                                                MPC[nx] += 1
                                                if MPC[nx] == 1:
                                                    PropertiesMPC(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    MPCTemp[nx] += 1
                                            # Ice-only
                                            elif chkICE>=1 and chkSCWice==0 and chkWARM==0 and chkSCW==0 :
                                                IceCloud[nx] += 1
                                                if IceCloud[nx] == 1:
                                                    PropertiesIceCloud(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    IceCloudTemp[nx] += 1
                                            # USLC
                                            elif chkSCW>=1 and chkICE>=0 and chkSCWice==0 and chkWARM==0 :
                                                USLC[nx] += 1
                                                if USLC[nx] == 1:
                                                    PropertiesUSLC(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    USLCTemp[nx] += 1
                                            # USLC with warm
                                            elif chkSCW>=1 and chkWARM>=1 and chkICE==0 and chkSCWice==0 :
                                                USLC2[nx] += 1
                                                if USLC2[nx] == 1:
                                                    PropertiesUSLC2(nx,topcloud,chkcloud,remainCLEAR)
                                                if tempe[nx,topcloud] <= 273.15:
                                                    USLC2Temp[nx] += 1
                                        # Warm-liquid cloud
                                        elif chkWARM>=0 and chkICE==1 and chkSCWice==0 and chkSCW==0 :
                                            WarmCloud[nx] += 1
                                            if WarmCloud[nx] == 1:
                                                PropertiesWarmCloud(nx,topcloud,chkcloud,remainCLEAR)
                                    
                                    # Precipitations
                                    if chkprecip >= 3:
                                        Precip[nx] += 1
                                        if Precip[nx] == 1:
                                            PrecipMatrix[nx] = 1
                                    
                                    #Tests
                                    if chkcoldR >= 3:
                                        ColdRain[nx] += 1
                                        if ColdRain[nx] == 1:
                                            ColdRainMatrix[nx] = 1
                                    if chkColdPrecip >= 3:
                                        ColdPrecip[nx] += 1
                                        if ColdPrecip[nx] == 1:
                                            ColdPrecipMatrix[nx] = 1
                                    if chkWarmPrecip >= 3:
                                        WarmPrecip[nx] += 1
                                        if WarmPrecip[nx] == 1:
                                            WarmPrecipMatrix[nx] = 1
                                    
                                    # Re-initialisation of counters
                                    chkSCWice = 0
                                    chkICE = 0
                                    chkWARM = 0
                                    chkSCW = 0
                                    chkCLEAR = 0
                                    remainCLEAR = 0
                                    chkcloud = 0
                                    topcloud = 0
                                    chkprecip = 0
                                    chkcoldR = 0
                                    chkColdPrecip = 0
                                    chkWarmPrecip = 0
                            
                                if MeteoConditions==True and TotalClouds[nx]>=1 and chkmeteo==0 :
                                    """ The following lines extract thermodynamic variables.
                                    
                                    The altitude range is not limited to 500-3000m to extract data outside of these boundaries.
                                    
                                    Please note that this part is only read when a cloud is detected,
                                    to only have data for cloud detections.
                                    """
                                    
                                    chkmeteo = 1
                                    
                                    # 700-mb level
                                    for nz2 in range (len(altiFull)-1,-1,-1):
                                        if pressure[nx,nz2] <= 70000:
                                            geopotential_700[nx] = altiFull[nz2]
                                            spec_hum_700[nx] = spec_hum[nx,nz2]
                                            temperature_700[nx] = temperatureFull[nx,nz2]
                                            break
                                    # 850-mb level
                                    for nz2 in range (len(altiFull)-1,-1,-1):
                                        if pressure[nx,nz2] <= 85000:
                                            geopotential_850[nx] = altiFull[nz2]
                                            spec_hum_850[nx] = spec_hum[nx,nz2]
                                            temperature_850[nx] = temperatureFull[nx,nz2]
                                            break
                                    # For integrated specific humidity between surface and 1500meters
                                    for nz2 in range (len(altiFull)-1,-1,-1):
                                        if altiFull[nz2] <= 0:
                                            SH500m[nx] = spec_hum[nx,nz2]
                                            pix500[nx] = 1
                                            SH1000m[nx] = spec_hum[nx,nz2]
                                            pix1000[nx] = 1
                                            SH1500m[nx] = spec_hum[nx,nz2]
                                            pix1500[nx] = 1
                                        elif altiFull[nz2] <= 0.5:
                                            SH500m[nx] += spec_hum[nx,nz2]
                                            pix500[nx] += 1
                                            SH1000m[nx] += spec_hum[nx,nz2]
                                            pix1000[nx] += 1
                                            SH1500m[nx] += spec_hum[nx,nz2]
                                            pix1500[nx] += 1
                                        elif altiFull[nz2] <= 1:
                                            SH1000m[nx] += spec_hum[nx,nz2]
                                            pix1000[nx] += 1
                                            SH1500m[nx] += spec_hum[nx,nz2]
                                            pix1500[nx] += 1
                                        elif altiFull[nz2] <= 1.5:
                                            SH1500m[nx] += spec_hum[nx,nz2]
                                        elif altiFull[nz2] > 1.5:
                                            break
                                    
                                    # Surface temperature
                                    if SST[nx] > 0:
                                        # SST for water surfaces
                                        SST_Skin_mask[nx] = SST[nx]
                                    else :
                                        # Skin temperature for land surfaces
                                        SST_Skin_mask[nx] = SkinTemp[nx]
                                    
                                    # Surface air temperature
                                    surf_temp[nx] = tempe2m[nx]
                                    
                                    # Surface pressure
                                    surf_press_mask[nx] = surf_press[nx]
                                    
                            # Defining Single and Multi clouds masks
                            if TotalClouds[nx] == 1:
                                SingleClouds[nx] = 1
                            
                            if TotalClouds[nx] >= 2:
                                MultiClouds[nx] = 1
                        
                        # Extraction of wind fields
                        if Winds == True:
                            # 700-mb level
                            for nz2 in range (len(altiFull)-1,-1,-1):
                                if pressure[nx,nz2] <= 70000:
                                    u_winds_700[nx] = Uwind[nx,nz2]
                                    v_winds_700[nx] = Vwind[nx,nz2]
                                    break
                            # 850-mb level
                            for nz2 in range (len(altiFull)-1,-1,-1):
                                if pressure[nx,nz2] <= 85000:
                                    u_winds_850[nx] = Uwind[nx,nz2]
                                    v_winds_850[nx] = Vwind[nx,nz2]
                                    break
                      
                  
                """STEP 5 - Creation of arrays for output textfiles
                
                """
                if not 'dardarPix_in_grid_Count' in locals():
                    dardarPix_in_grid_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                # GM masks
                if not 'pix_in_cloud_Count' in locals():
                    pix_in_cloud_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if not 'Allpix_in_cloud_Count' in locals():
                    Allpix_in_cloud_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if not 'Allpix_in_cloud_occur_Count' in locals():
                    Allpix_in_cloud_occur_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if not 'cloud_occurence_in_pix_Count' in locals():
                    cloud_occurence_in_pix_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                if AllScooled_mask==True and not 'allSupercooled_Count' in locals():
                    allSupercooled_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if AllScooled_mask==True and not 'allSupercooled2_Count' in locals():
                    allSupercooled2_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if IceAndSupercooled_mask==True and not 'iceAndSupercooled_Count' in locals():
                    iceAndSupercooled_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if ScooledOnly_mask==True and not 'supercooledOnly_Count' in locals():
                    supercooledOnly_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if ScooledOnlyMPC_mask==True and not 'supercooledOnlyMPC_Count' in locals():
                    supercooledOnlyMPC_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if ScooledOnlyMPC_mask==True and not 'supercooledOnlyALONE_Count' in locals():
                    supercooledOnlyALONE_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if IceOnly_mask==True and not 'iceOnly_Count' in locals():
                    iceOnly_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if IceContaining_cloud_1_mask==True and not 'iceContaining_cloud_1_Count' in locals():
                    iceContaining_cloud_1_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if IceContaining_cloud_2_mask==True and not 'iceContaining_cloud_2_Count' in locals():
                    iceContaining_cloud_2_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if liquidWarm_mask==True and not 'liqWarm_Count' in locals():
                    liqWarm_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if classif_mix_clouds == True :
                    if not 'mask_cloud_sc_ice_BelowAbove_Count' in locals():
                        mask_cloud_sc_ice_BelowAbove_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'mask_cloud_sc_ice_BelowOnly_Count' in locals():
                        mask_cloud_sc_ice_BelowOnly_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'mask_cloud_sc_ice_BelowAbove2_Count' in locals():
                        mask_cloud_sc_ice_BelowAbove2_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'mask_cloud_sc_ice_BelowOnly2_Count' in locals():
                        mask_cloud_sc_ice_BelowOnly2_Count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                # Thermodynamic variables
                if temperature==True and not 'tempe_sum' in locals():
                    tempe_sum = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if temperature_2m==True and not 'tempe2m_sum' in locals():
                    tempe2m_sum = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if iwc_val==True and not 'iwc_sum' in locals():
                    iwc_sum = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if ext_val==True and not 'ext_sum' in locals():
                    ext_sum = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)        
                if re_val==True and not 're_sum' in locals():
                    re_sum = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                if lidarRatio_val==True and not 'LR_sum' in locals():
                    LR_sum = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                # CB masks
                
                if CloudSatPrecip == True :
                    if not 'CSRain_count' in locals():
                        CSRain_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'CSSnow_count' in locals():
                        CSSnow_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'CSMixed_count' in locals():
                        CSMixed_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'CSAll_count' in locals():
                        CSAll_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                if Surface_Type == True : 
                    if not 'Surface_Land_count' in locals():
                        Surface_Land_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'Surface_Water_count' in locals():
                        Surface_Water_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'Surface_Ice_count' in locals():
                        Surface_Ice_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float) 
                        
                
                if CloudAnalysis == True :
                    if not 'Cloud_Between_Lidar_count' in locals():
                        Cloud_Between_Lidar_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'MPC_count' in locals():
                        MPC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'MPCTemp_count' in locals():
                        MPCTemp_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'USLC_count' in locals():
                        USLC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'USLC2_count' in locals():
                        USLC2_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'USLCTemp_count' in locals():
                        USLCTemp_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'USLC2Temp_count' in locals():
                        USLC2Temp_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'IC_count' in locals():
                        IC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ICTemp_count' in locals():
                        ICTemp_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'WC_count' in locals():
                        WC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'CC_count' in locals():
                        CC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'CCMatrix_count' in locals():
                        CCMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TotalClouds_count' in locals():
                        TotalClouds_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TotalCloudsMatrix_count' in locals():
                        TotalCloudsMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'SingleClouds_count' in locals():
                        SingleClouds_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'MultiClouds_count' in locals():
                        MultiClouds_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                    if not 'MPCMatrix_count' in locals():
                        MPCMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'HeightMPC_count' in locals():
                        HeightMPC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TempMPC_count' in locals():
                        TempMPC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ThicknessMPC_count' in locals():
                        ThicknessMPC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                    if not 'USLCMatrix_count' in locals():
                        USLCMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'HeightUSLC_count' in locals():
                        HeightUSLC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TempUSLC_count' in locals():
                        TempUSLC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ThicknessUSLC_count' in locals():
                        ThicknessUSLC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                    if not 'USLC2Matrix_count' in locals():
                        USLC2Matrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'HeightUSLC2_count' in locals():
                        HeightUSLC2_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TempUSLC2_count' in locals():
                        TempUSLC2_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ThicknessUSLC2_count' in locals():
                        ThicknessUSLC2_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                
                    if not 'ICMatrix_count' in locals():
                        ICMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'HeightIC_count' in locals():
                        HeightIC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TempIC_count' in locals():
                        TempIC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ThicknessIC_count' in locals(): 
                        ThicknessIC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    
                    if not 'WCMatrix_count' in locals():
                        WCMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'HeightWC_count' in locals():
                        HeightWC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'TempWC_count' in locals():
                        TempWC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ThicknessWC_count' in locals():
                        ThicknessWC_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    
                    if not 'Precip_count' in locals():
                        Precip_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'PrecipMatrix_count' in locals():
                        PrecipMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ColdRain_count' in locals():
                        ColdRain_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ColdRainMatrix_count' in locals():
                        ColdRainMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    
                    if not 'ColdPrecip_count' in locals():
                        ColdPrecip_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'ColdPrecipMatrix_count' in locals():
                        ColdPrecipMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'WarmPrecip_count' in locals():
                        WarmPrecip_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    if not 'WarmPrecipMatrix_count' in locals():
                        WarmPrecipMatrix_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    
                    if MeteoConditions==True :
                        if not 'geopotential_700_count' in locals():
                            geopotential_700_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'spec_hum_700_count' in locals():
                            spec_hum_700_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'temperature_700_count' in locals():
                            temperature_700_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'geopotential_850_count' in locals():
                            geopotential_850_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'spec_hum_850_count' in locals():
                            spec_hum_850_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'temperature_850_count' in locals():
                            temperature_850_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'SST_Skin_count' in locals():
                            SST_Skin_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'surf_temp_count' in locals():
                            surf_temp_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'surf_press_count' in locals():
                            surf_press_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        
                        if not 'SH500m_count' in locals():
                            SH500m_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'SH1000m_count' in locals():
                            SH1000m_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'SH1500m_count' in locals():
                            SH1500m_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'pix500_count' in locals():
                            pix500_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'pix1000_count' in locals():
                            pix1000_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'pix1500_count' in locals():
                            pix1500_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    
                    if Winds==True:
                        if not 'u_winds_700_count' in locals():
                            u_winds_700_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'v_winds_700_count' in locals():
                            v_winds_700_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'u_winds_850_count' in locals():
                            u_winds_850_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'v_winds_850_count' in locals():
                            v_winds_850_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'Surface_Uwind_count' in locals():
                            Surface_Uwind_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                        if not 'Surface_Vwind_count' in locals():
                            Surface_Vwind_count = array(np.zeros(nbLat*nbLong).reshape((nbLat,nbLong)),float)
                    
                   
                """STEP 6 - Filling the arrays with previously extracted data.
                
                """
                
                for nlat in range(len(rangeLat)-1):
                    for nlong in range(len(rangeLong)-1):
                        
                        # Defining the grid box
                        idx_deltaLat = []  # Used for test condtions, otherwise this is an array.
                        chk = np.copy(lat)
                        id1 = ((chk<float(rangeLat[nlat+1])) & (chk>=float(rangeLat[nlat]))).nonzero()
                        chk2 = np.copy(lon)
                        id2 = ((chk2<float(rangeLong[nlong+1])) & (chk2>=float(rangeLong[nlong]))).nonzero()
                        idx_deltaLat = array(list(set((id1[0])).intersection(set((id2[0])))))
                        
                        
                        # Surface condition 
                                    
                        if idx_deltaLat != []:
                            if landWaterFlag != None:
                                if landWaterFlag == "water":
                                    idx_deltaLat2 = idx_deltaLat[((landWater_flag[idx_deltaLat]==6)\
                                    | (landWater_flag[idx_deltaLat]==7) | (landWater_flag[idx_deltaLat]==0)).nonzero()]
                                
                                if landWaterFlag == "land":
                                    idx_deltaLat2 = idx_deltaLat[((landWater_flag[idx_deltaLat]!=6)\
                                    & (landWater_flag[idx_deltaLat]!=7) & (landWater_flag[idx_deltaLat]!=0)).nonzero()]
                            else:
                                idx_deltaLat2 = idx_deltaLat
                        
                        del chk,chk2
                        
                        # Day/Night condition
                        if idx_deltaLat != []:
                            if dayNightFlag != None:
                                if dayNightFlag == "day":
                                    idx_deltaLat2 = idx_deltaLat[(dayNight_flag[idx_deltaLat]==0).nonzero()]
                            
                            else:
                                idx_deltaLat2 = idx_deltaLat
                        
                        
                        # Filling the grid box with data
                        if idx_deltaLat != [] and idx_deltaLat2 != []:
                            
                            if Surface_Type == True: 
                                if not np.isnan(nansum(land_surface_mask[idx_deltaLat2])):
                                    Surface_Land_count[nlat,nlong] += nansum(land_surface_mask[idx_deltaLat2])
                                if not np.isnan(nansum(water_surface_mask[idx_deltaLat2])):
                                    Surface_Water_count[nlat,nlong] += nansum(water_surface_mask[idx_deltaLat2])
                                if not np.isnan(nansum(ice_surface_mask[idx_deltaLat2])):
                                    Surface_Ice_count[nlat,nlong] += nansum(ice_surface_mask[idx_deltaLat2])

                            dardarPix_in_grid_Count[nlat,nlong] += float(len(idx_deltaLat2))                                                                                 
                            pix_in_cloud_Count[nlat,nlong] += float(nansum(pix_in_cloud[idx_deltaLat2,:]))
                            
                            # For GM only
                            chk = np.copy(nansum(pix_in_cloud[idx_deltaLat2,:],1))
                            chk[(chk<3)] = 0.
                            chk[(chk>=3)]  = 1.
                            cloud_occurence_in_pix_Count[nlat,nlong] += nansum(chk)                                                      
                            del chk
                            # For GM only
                            Allpix_in_cloud_Count[nlat,nlong] += float(nansum(Allpix_in_cloud[idx_deltaLat2,:]))
                            # For GM only
                            chk = np.copy(nansum(Allpix_in_cloud[idx_deltaLat2,:],1))
                            chk[(chk<3)] = 0.
                            chk[(chk>=3)]  = 1.
                            Allpix_in_cloud_occur_Count[nlat,nlong] += nansum(chk)
                            del chk
                            
                            # IceAndSC_GEO file
                            if IceAndSupercooled_mask == True:                           
                                if float(nansum(mask_cloud_iceAndSupercooled[idx_deltaLat2,:])) != 0.:                                        
                                    chk = np.copy(nansum(mask_cloud_iceAndSupercooled[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    iceAndSupercooled_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    iceAndSupercooled_Count[nlat,nlong] += 0.                                
                            
                            if AllScooled_mask == True:
                                #AllSCGEO file
                                if float(nansum(mask_cloud_allSupercooled[idx_deltaLat2,:])) != 0.:                                        
                                    chk = np.copy(nansum(mask_cloud_allSupercooled[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    allSupercooled_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    allSupercooled_Count[nlat,nlong] += 0.                                
                                #AllSC2GEO file
                                if float(nansum(mask_cloud_allSupercooled2[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_allSupercooled2[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    allSupercooled2_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    allSupercooled2_Count[nlat,nlong] += 0.  
                            
                            #SCOnlyGEO file
                            if ScooledOnly_mask == True:  
                                if float(nansum(mask_cloud_supercooledOnly[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_supercooledOnly[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    supercooledOnly_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    supercooledOnly_Count[nlat,nlong] += 0.
                            
                            if ScooledOnlyMPC_mask == True:
                                #SCOnlyMPCGEO file
                                if float(nansum(mask_cloud_supercooledOnlyMPC[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_supercooledOnlyMPC[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    supercooledOnlyMPC_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    supercooledOnlyMPC_Count[nlat,nlong] += 0.
                                #SCOnlyALONEGEO file
                                if float(nansum(mask_cloud_supercooledOnlyALONE[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_supercooledOnlyALONE[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    supercooledOnlyALONE_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    supercooledOnlyALONE_Count[nlat,nlong] += 0.
                            
                            #IceOnlyGEO file
                            if IceOnly_mask == True:  
                                if float(nansum(mask_cloud_iceOnly[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_iceOnly[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    iceOnly_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    iceOnly_Count[nlat,nlong] += 0.                                                                                               
                            
                            #IceContainingCloud1_GEO file
                            if IceContaining_cloud_1_mask == True:  
                                if float(nansum(mask_ice_containing_cloud_1[idx_deltaLat2,:])) != 0.:                                        
                                    chk = np.copy(nansum(mask_ice_containing_cloud_1[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    iceContaining_cloud_1_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    iceContaining_cloud_1_Count[nlat,nlong] += 0.
                            
                            #IceContainingCloud2_GEO file
                            if IceContaining_cloud_2_mask == True:  
                                if float(nansum(mask_ice_containing_cloud_2[idx_deltaLat2,:])) !=0.:                                        
                                    chk = np.copy(nansum(mask_ice_containing_cloud_2[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    iceContaining_cloud_2_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    iceContaining_cloud_1_Count[nlat,nlong] += 0.
                            
                            #LiquidWarmGEO file
                            if liquidWarm_mask == True:  
                                if float(nansum(mask_cloud_liqWarm[idx_deltaLat2,:])) !=0.:                                        
                                    chk=np.copy(nansum(mask_cloud_liqWarm[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0.
                                    chk[(chk>=3)] = 1.
                                    liqWarm_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    liqWarm_Count[nlat,nlong] += 0.   
                                    
                            if classif_mix_clouds == True :
                                #SC_ice_belowAboveGEO file
                                if float(nansum(mask_cloud_sc_ice_BelowAbove[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_sc_ice_BelowAbove[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0
                                    chk[(chk>=3)] = 1
                                    mask_cloud_sc_ice_BelowAbove_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    mask_cloud_sc_ice_BelowAbove_Count[nlat,nlong] += 0.
                                #SC_ice_belowOnlyGEO file
                                if float(nansum(mask_cloud_sc_ice_BelowOnly[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_sc_ice_BelowOnly[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0
                                    chk[(chk>=3)] = 1
                                    mask_cloud_sc_ice_BelowOnly_Count[nlat,nlong] += nansum(chk)                                  
                                    del chk
                                else:
                                    mask_cloud_sc_ice_BelowOnly_Count[nlat,nlong] += 0.
                                #SC_ice_belowAbove2GEO file
                                if float(nansum(mask_cloud_sc_ice_BelowAbove2[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_sc_ice_BelowAbove2[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0
                                    chk[(chk>=3)] = 1
                                    mask_cloud_sc_ice_BelowAbove2_Count[nlat,nlong] += nansum(chk)
                                    del chk                                  
                                else:
                                    mask_cloud_sc_ice_BelowAbove2_Count[nlat,nlong] += 0.
                                #SC_ice_belowOnly2GEO file
                                if float(nansum(mask_cloud_sc_ice_BelowOnly2[idx_deltaLat2,:])) != 0.:                                        
                                    chk=np.copy(nansum(mask_cloud_sc_ice_BelowOnly2[idx_deltaLat2,:],1))
                                    chk[(chk<3)] = 0
                                    chk[(chk>=3)] = 1
                                    mask_cloud_sc_ice_BelowOnly2_Count[nlat,nlong] += nansum(chk)                                  
                                    del chk
                                else:
                                    mask_cloud_sc_ice_BelowOnly2_Count[nlat,nlong] += 0.
                            
                            # Physical parameters
                            if temperature == True and (not np.isnan(nanmean(nanmean(tempe[idx_deltaLat2,:])))):            
                                tempe_sum[nlat,nlong] += nansum(tempe[idx_deltaLat2,:])
                            if temperature_2m == True and (not np.isnan(nanmean(tempe2m[idx_deltaLat2]))):
                                tempe2m_sum[nlat,nlong] += nansum(tempe2m[idx_deltaLat2])
                            if iwc_val == True and (not np.isnan(nanmean(nanmean(iwc[idx_deltaLat2,:])*1000.))):
                                iwc_sum[nlat,nlong] += nansum(iwc[idx_deltaLat2,:])*1000.     ## kg/m3 --> g/m3
                            if ext_val == True and (not np.isnan(nanmean(nanmean(ext[idx_deltaLat2,:])*1000.))):
                                ext_sum[nlat,nlong] += nansum(ext[idx_deltaLat2,:])*1000.     ## m-1 --> km-1
                            if re_val == True and (not np.isnan(nanmean(nanmean(re[idx_deltaLat2,:])*1000000.))):
                                re_sum[nlat,nlong] += nansum(re[idx_deltaLat2,:])*1000000.   ## m --> micrometers:           
                            if lidarRatio_val == True and (not np.isnan(nanmean(nanmean(LR[idx_deltaLat2,:])))):
                                LR_sum[nlat,nlong] += nansum(LR[idx_deltaLat2,:])
                            
                            # CloudSat Precip
                            if CloudSatPrecip == True :
                                if not np.isnan(nansum(CSRain[idx_deltaLat2])):
                                    CSRain_count[nlat,nlong] += nansum(CSRain[idx_deltaLat2])
                                if not np.isnan(nansum(CSSnow[idx_deltaLat2])):
                                    CSSnow_count[nlat,nlong] += nansum(CSSnow[idx_deltaLat2])
                                if not np.isnan(nansum(CSMixed[idx_deltaLat2])):
                                    CSMixed_count[nlat,nlong] += nansum(CSMixed[idx_deltaLat2])
                                if not np.isnan(nansum(CSAll[idx_deltaLat2])):
                                    CSAll_count[nlat,nlong] += nansum(CSAll[idx_deltaLat2])
                            
                        
                            # CB analysis
                            if CloudAnalysis == True:
                                if not np.isnan(nansum(LidarExtCloudMatrix[idx_deltaLat2])):
                                    Cloud_Between_Lidar_count[nlat,nlong] += nansum(LidarExtCloudMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(MPC[idx_deltaLat2])):
                                    MPC_count[nlat,nlong] += nansum(MPC[idx_deltaLat2])
                                if not np.isnan(nansum(MPCTemp[idx_deltaLat2])):
                                    MPCTemp_count[nlat,nlong] += nansum(MPCTemp[idx_deltaLat2])
                                if not np.isnan(nansum(USLC[idx_deltaLat2])):
                                    USLC_count[nlat,nlong] += nansum(USLC[idx_deltaLat2])
                                if not np.isnan(nansum(USLC2[idx_deltaLat2])):
                                    USLC2_count[nlat,nlong] += nansum(USLC2[idx_deltaLat2])
                                if not np.isnan(nansum(USLCTemp[idx_deltaLat2])):
                                    USLCTemp_count[nlat,nlong] += nansum(USLCTemp[idx_deltaLat2])
                                if not np.isnan(nansum(USLC2Temp[idx_deltaLat2])):
                                    USLC2Temp_count[nlat,nlong] += nansum(USLC2Temp[idx_deltaLat2])
                                if not np.isnan(nansum(IceCloud[idx_deltaLat2])):
                                    IC_count[nlat,nlong] += nansum(IceCloud[idx_deltaLat2])
                                if not np.isnan(nansum(IceCloudTemp[idx_deltaLat2])):
                                    ICTemp_count[nlat,nlong] += nansum(IceCloudTemp[idx_deltaLat2])
                                if not np.isnan(nansum(WarmCloud[idx_deltaLat2])):
                                    WC_count[nlat,nlong] += nansum(WarmCloud[idx_deltaLat2])
                                if not np.isnan(nansum(ColdCloud[idx_deltaLat2])):
                                    CC_count[nlat,nlong] += nansum(ColdCloud[idx_deltaLat2])
                                if not np.isnan(nansum(ColdCloudMatrix[idx_deltaLat2])):
                                    CCMatrix_count[nlat,nlong] += nansum(ColdCloudMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(TotalClouds[idx_deltaLat2])):
                                    TotalClouds_count[nlat,nlong] += nansum(TotalClouds[idx_deltaLat2])
                                if not np.isnan(nansum(TotalCloudsMatrix[idx_deltaLat2])):
                                    TotalCloudsMatrix_count[nlat,nlong] += nansum(TotalCloudsMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(SingleClouds[idx_deltaLat2])):
                                    SingleClouds_count[nlat,nlong] += nansum(SingleClouds[idx_deltaLat2])
                                if not np.isnan(nansum(MultiClouds[idx_deltaLat2])):
                                    MultiClouds_count[nlat,nlong] += nansum(MultiClouds[idx_deltaLat2])
                                
                                if not np.isnan(nansum(MPCMatrix[idx_deltaLat2])):
                                    MPCMatrix_count[nlat,nlong] += nansum(MPCMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(TopHeightMPC[idx_deltaLat2])):
                                    HeightMPC_count[nlat,nlong] += nansum(TopHeightMPC[idx_deltaLat2])
                                if not np.isnan(nansum(TopTempMPC[idx_deltaLat2])):
                                    TempMPC_count[nlat,nlong] += nansum(TopTempMPC[idx_deltaLat2])
                                if not np.isnan(nansum(ThicknessMPC[idx_deltaLat2])):
                                    ThicknessMPC_count[nlat,nlong] += nansum(ThicknessMPC[idx_deltaLat2])
                                
                                if not np.isnan(nansum(USLCMatrix[idx_deltaLat2])):
                                    USLCMatrix_count[nlat,nlong] += nansum(USLCMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(TopHeightUSLC[idx_deltaLat2])):
                                    HeightUSLC_count[nlat,nlong] += nansum(TopHeightUSLC[idx_deltaLat2])
                                if not np.isnan(nansum(TopTempUSLC[idx_deltaLat2])):
                                    TempUSLC_count[nlat,nlong] += nansum(TopTempUSLC[idx_deltaLat2])
                                if not np.isnan(nansum(ThicknessUSLC[idx_deltaLat2])):
                                    ThicknessUSLC_count[nlat,nlong] += nansum(ThicknessUSLC[idx_deltaLat2])
                                    
                                if not np.isnan(nansum(USLC2Matrix[idx_deltaLat2])):
                                    USLC2Matrix_count[nlat,nlong] += nansum(USLC2Matrix[idx_deltaLat2])
                                if not np.isnan(nansum(TopHeightUSLC[idx_deltaLat2])):
                                    HeightUSLC2_count[nlat,nlong] += nansum(TopHeightUSLC2[idx_deltaLat2])
                                if not np.isnan(nansum(TopTempUSLC2[idx_deltaLat2])):
                                    TempUSLC2_count[nlat,nlong] += nansum(TopTempUSLC2[idx_deltaLat2])
                                if not np.isnan(nansum(ThicknessUSLC2[idx_deltaLat2])):
                                    ThicknessUSLC2_count[nlat,nlong] += nansum(ThicknessUSLC2[idx_deltaLat2])
                                
                                if not np.isnan(nansum(IceCloudMatrix[idx_deltaLat2])):
                                    ICMatrix_count[nlat,nlong] += nansum(IceCloudMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(TopHeightIceCloud[idx_deltaLat2])):
                                    HeightIC_count[nlat,nlong] += nansum(TopHeightIceCloud[idx_deltaLat2])
                                if not np.isnan(nansum(TopTempIceCloud[idx_deltaLat2])):
                                    TempIC_count[nlat,nlong] += nansum(TopTempIceCloud[idx_deltaLat2])
                                if not np.isnan(nansum(ThicknessIceCloud[idx_deltaLat2])):
                                    ThicknessIC_count[nlat,nlong] += nansum(ThicknessIceCloud[idx_deltaLat2])
                                
                                if not np.isnan(nansum(WarmCloudMatrix[idx_deltaLat2])):
                                    WCMatrix_count[nlat,nlong] += nansum(WarmCloudMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(TopHeightWarmCloud[idx_deltaLat2])):
                                    HeightWC_count[nlat,nlong] += nansum(TopHeightWarmCloud[idx_deltaLat2])
                                if not np.isnan(nansum(TopTempWarmCloud[idx_deltaLat2])):
                                    TempWC_count[nlat,nlong] += nansum(TopTempWarmCloud[idx_deltaLat2])
                                if not np.isnan(nansum(ThicknessWarmCloud[idx_deltaLat2])):
                                    ThicknessWC_count[nlat,nlong] += nansum(ThicknessWarmCloud[idx_deltaLat2])
                                
                                # DARDAR Precip
                                if not np.isnan(nansum(Precip[idx_deltaLat2])):
                                    Precip_count[nlat,nlong] += nansum(Precip[idx_deltaLat2])
                                if not np.isnan(nansum(PrecipMatrix[idx_deltaLat2])):
                                    PrecipMatrix_count[nlat,nlong] += nansum(PrecipMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(ColdRain[idx_deltaLat2])):
                                    ColdRain_count[nlat,nlong] += nansum(ColdRain[idx_deltaLat2])
                                if not np.isnan(nansum(ColdRainMatrix[idx_deltaLat2])):
                                    ColdRainMatrix_count[nlat,nlong] += nansum(ColdRainMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(ColdPrecip[idx_deltaLat2])):
                                    ColdPrecip_count[nlat,nlong] += nansum(ColdPrecip[idx_deltaLat2])
                                if not np.isnan(nansum(ColdPrecipMatrix[idx_deltaLat2])):
                                    ColdPrecipMatrix_count[nlat,nlong] += nansum(ColdPrecipMatrix[idx_deltaLat2])
                                if not np.isnan(nansum(WarmPrecip[idx_deltaLat2])):
                                    WarmPrecip_count[nlat,nlong] += nansum(WarmPrecip[idx_deltaLat2])
                                if not np.isnan(nansum(WarmPrecipMatrix[idx_deltaLat2])):
                                    WarmPrecipMatrix_count[nlat,nlong] += nansum(WarmPrecipMatrix[idx_deltaLat2])
                                    
                            if MeteoConditions == True:
                                if not np.isnan(nansum(geopotential_700[idx_deltaLat2])):
                                    geopotential_700_count[nlat,nlong] += nansum(geopotential_700[idx_deltaLat2])
                                if not np.isnan(nansum(spec_hum_700[idx_deltaLat2])):
                                    spec_hum_700_count[nlat,nlong] += nansum(spec_hum_700[idx_deltaLat2])
                                if not np.isnan(nansum(temperature_700[idx_deltaLat2])):
                                    temperature_700_count[nlat,nlong] += nansum(temperature_700[idx_deltaLat2])
                                if not np.isnan(nansum(geopotential_850[idx_deltaLat2])):
                                    geopotential_850_count[nlat,nlong] += nansum(geopotential_850[idx_deltaLat2])
                                if not np.isnan(nansum(spec_hum_850[idx_deltaLat2])):
                                    spec_hum_850_count[nlat,nlong] += nansum(spec_hum_850[idx_deltaLat2])
                                if not np.isnan(nansum(temperature_850[idx_deltaLat2])):
                                    temperature_850_count[nlat,nlong] += nansum(temperature_850[idx_deltaLat2])
                                if not np.isnan(nansum(SST_Skin_mask[idx_deltaLat2])):
                                    SST_Skin_count[nlat,nlong] += nansum(SST_Skin_mask[idx_deltaLat2])
                                if not np.isnan(nansum(surf_temp[idx_deltaLat2])):
                                    surf_temp_count[nlat,nlong] += nansum(surf_temp[idx_deltaLat2])
                                if not np.isnan(nansum(surf_press_mask[idx_deltaLat2])):
                                    surf_press_count[nlat,nlong] += nansum(surf_press_mask[idx_deltaLat2])
                                
                                # Not used
                                if not np.isnan(nansum(SH500m[idx_deltaLat2])):
                                    SH500m_count[nlat,nlong] += nansum(SH500m[idx_deltaLat2])
                                if not np.isnan(nansum(SH1000m[idx_deltaLat2])):
                                    SH1000m_count[nlat,nlong] += nansum(SH1000m[idx_deltaLat2])
                                if not np.isnan(nansum(SH1500m[idx_deltaLat2])):
                                    SH1500m_count[nlat,nlong] += nansum(SH1500m[idx_deltaLat2])
                                if not np.isnan(nansum(pix500[idx_deltaLat2])):
                                    pix500_count[nlat,nlong] += nansum(pix500[idx_deltaLat2])
                                if not np.isnan(nansum(pix1000[idx_deltaLat2])):
                                    pix1000_count[nlat,nlong] += nansum(pix1000[idx_deltaLat2])
                                if not np.isnan(nansum(pix1500[idx_deltaLat2])):
                                    pix1500_count[nlat,nlong] += nansum(pix1500[idx_deltaLat2])
                                    
                            if Winds == True:
                                if not np.isnan(nansum(u_winds_700[idx_deltaLat2])):
                                    u_winds_700_count[nlat,nlong] += nansum(u_winds_700[idx_deltaLat2])
                                if not np.isnan(nansum(v_winds_700[idx_deltaLat2])):
                                    v_winds_700_count[nlat,nlong] += nansum(v_winds_700[idx_deltaLat2])
                                if not np.isnan(nansum(u_winds_850[idx_deltaLat2])):
                                    u_winds_850_count[nlat,nlong] += nansum(u_winds_850[idx_deltaLat2])
                                if not np.isnan(nansum(v_winds_850[idx_deltaLat2])):
                                    v_winds_850_count[nlat,nlong] += nansum(v_winds_850[idx_deltaLat2])
                                if not np.isnan(nansum(Surface_Uwind[idx_deltaLat2])):
                                    Surface_Uwind_count[nlat,nlong] += nansum(Surface_Uwind[idx_deltaLat2])
                                if not np.isnan(nansum(Surface_Vwind[idx_deltaLat2])):
                                    Surface_Vwind_count[nlat,nlong] += nansum(Surface_Vwind[idx_deltaLat2])

                           
                        else:
                            dardarPix_in_grid_Count[nlat,nlong] += 0.                            
                            pix_in_cloud_Count[nlat,nlong] += 0.
                            Allpix_in_cloud_Count[nlat,nlong] += 0.
                            Allpix_in_cloud_occur_Count[nlat,nlong] += 0.
                            cloud_occurence_in_pix_Count[nlat,nlong] += 0.
                            
                            if IceAndSupercooled_mask == True:
                                iceAndSupercooled_Count[nlat,nlong] += 0.
                            if AllScooled_mask == True:
                                allSupercooled_Count[nlat,nlong] += 0.
                            if AllScooled_mask == True:
                                allSupercooled2_Count[nlat,nlong] += 0.
                            if ScooledOnly_mask == True:
                                supercooledOnly_Count[nlat,nlong] += 0.
                            if ScooledOnlyMPC_mask == True:
                                supercooledOnlyMPC_Count[nlat,nlong] += 0.
                            if ScooledOnlyMPC_mask == True:
                                supercooledOnlyALONE_Count[nlat,nlong] += 0.
                            if IceOnly_mask == True:
                                iceOnly_Count[nlat,nlong] += 0.
                            if IceContaining_cloud_1_mask == True:
                                iceContaining_cloud_1_Count[nlat,nlong] += 0.
                            if IceContaining_cloud_2_mask == True:
                                iceContaining_cloud_2_Count[nlat,nlong] += 0.
                            if liquidWarm_mask == True:
                                liqWarm_Count[nlat,nlong] += 0.
                            if classif_mix_clouds == True:
                                mask_cloud_sc_ice_BelowAbove_Count[nlat,nlong] += 0.
                                mask_cloud_sc_ice_BelowOnly_Count[nlat,nlong] += 0.
                                mask_cloud_sc_ice_BelowAbove2_Count[nlat,nlong] += 0.
                                mask_cloud_sc_ice_BelowOnly2_Count[nlat,nlong] += 0.
                                
                            if temperature == True:
                                tempe_sum[nlat,nlong] += 0.
                            if temperature_2m == True:
                                tempe2m_sum[nlat,nlong] += 0.
                            if iwc_val == True:
                                iwc_sum[nlat,nlong] += 0.
                            if ext_val == True:
                                ext_sum[nlat,nlong] += 0.
                            if re_val == True:
                                re_sum[nlat,nlong] += 0.
                            if lidarRatio_val == True:
                                LR_sum[nlat,nlong] += 0.
                            
                            if CloudSatPrecip == True:
                                CSRain_count[nlat,nlong] += 0.
                                CSSnow_count[nlat,nlong] += 0.
                                CSMixed_count[nlat,nlong] += 0.
                                CSAll_count[nlat,nlong] += 0.
                            
                            if CloudAnalysis == True:
                                Cloud_Between_Lidar_count[nlat,nlong] += 0.
                                MPC_count[nlat,nlong] += 0.
                                MPCTemp_count[nlat,nlong] += 0.
                                USLC_count[nlat,nlong] += 0.
                                USLC2_count[nlat,nlong] += 0.
                                USLCTemp_count[nlat,nlong] += 0.
                                USLC2Temp_count[nlat,nlong] += 0.
                                IC_count[nlat,nlong] += 0.
                                ICTemp_count[nlat,nlong] += 0.
                                WC_count[nlat,nlong] += 0.
                                CC_count[nlat,nlong] += 0.
                                CCMatrix_count[nlat,nlong] += 0.
                                TotalClouds_count[nlat,nlong] += 0.
                                TotalCloudsMatrix_count[nlat,nlong] += 0.
                                SingleClouds_count[nlat,nlong] += 0.
                                MultiClouds_count[nlat,nlong] += 0.
                            
                                MPCMatrix_count[nlat,nlong] += 0.
                                HeightMPC_count[nlat,nlong] += 0.
                                TempMPC_count[nlat,nlong] += 0.
                                ThicknessMPC_count[nlat,nlong] += 0.
                                
                                USLCMatrix_count[nlat,nlong] += 0.
                                HeightUSLC_count[nlat,nlong] += 0.
                                TempUSLC_count[nlat,nlong] += 0.
                                ThicknessUSLC_count[nlat,nlong] += 0.
                                
                                USLC2Matrix_count[nlat,nlong] += 0.
                                HeightUSLC2_count[nlat,nlong] += 0.
                                TempUSLC2_count[nlat,nlong] += 0.
                                ThicknessUSLC2_count[nlat,nlong] += 0.
                                
                                ICMatrix_count[nlat,nlong] += 0.
                                HeightIC_count[nlat,nlong] += 0.
                                TempIC_count[nlat,nlong] += 0.
                                ThicknessIC_count[nlat,nlong] += 0.
                                
                                WCMatrix_count[nlat,nlong] += 0.
                                HeightWC_count[nlat,nlong] += 0.
                                TempWC_count[nlat,nlong] += 0.
                                ThicknessWC_count[nlat,nlong] += 0.
                                
                                Precip_count[nlat,nlong] += 0.
                                PrecipMatrix_count[nlat,nlong] += 0.
                                ColdRain_count[nlat,nlong] += 0.
                                ColdRainMatrix_count[nlat,nlong] += 0.
                                
                                ColdPrecip_count[nlat,nlong] += 0.
                                ColdPrecipMatrix_count[nlat,nlong] += 0.
                                WarmPrecip_count[nlat,nlong] += 0.
                                WarmPrecipMatrix_count[nlat,nlong] += 0.
                            
                            if MeteoConditions == True:
                                geopotential_700_count[nlat,nlong] += 0.
                                spec_hum_700_count[nlat,nlong] += 0.
                                temperature_700_count[nlat,nlong] += 0.
                                geopotential_850_count[nlat,nlong] += 0.
                                spec_hum_850_count[nlat,nlong] += 0.
                                temperature_850_count[nlat,nlong] += 0.
                                SST_Skin_count[nlat,nlong] += 0.
                                surf_temp_count[nlat,nlong] += 0.
                                surf_press_count[nlat,nlong] += 0.
                                
                                SH500m_count[nlat,nlong] += 0.
                                SH1000m_count[nlat,nlong] += 0.
                                SH1500m_count[nlat,nlong] += 0.
                                pix500_count[nlat,nlong] += 0.
                                pix1000_count[nlat,nlong] += 0.
                                pix1500_count[nlat,nlong] += 0.
                            
                            if Winds == True:
                                u_winds_700_count[nlat,nlong] += 0.
                                v_winds_700_count[nlat,nlong] += 0.
                                u_winds_850_count[nlat,nlong] += 0.
                                v_winds_850_count[nlat,nlong] += 0.
                                Surface_Uwind_count[nlat,nlong] += 0.
                                Surface_Vwind_count[nlat,nlong] += 0.
                            
                      
                print "add to previous data OK"
                
            else:
                print "no latitude limits could be found ???"

        else: print "...........granule out of longitude limits"
        
        """STEP 7 : Segmentation for file creation
        Data is written in the file :
        - if 7 days have been processed and if there are other files to process
        - if the whole list of files have been processed
        - if 7 days have been processed and the following day is in the same month
        - if the following day is in another month
        """
        
        cond1, cond2, cond3 = 0., 0., 0.

        """ If 7 days have been processed and there is no month change next :
        Before Oct 2017:
            if (comptDay4segmentation!=0. and comptDay4segmentation%segFile==0.\
            and nHDFfiles!=len(listFINALE)-1\
            and int(listFINALE[nHDFfiles+1][-19:-16])!=julianDayGranule\
        Between Oct 2017 and Apr 2021 :
            if (comptDay4segmentation!=0. and comptDay4segmentation%segFile==0.\
            and nHDFfiles!=len(listFINALE)-1\
            and int(listFINALE[nHDFfiles+1][-25:-22])!=julianDayGranule\
        """
        # Since Apr 2021 : 
        if (comptDay4segmentation!=0. and comptDay4segmentation%segFile==0.\
            and nHDFfiles!=len(listFINALE)-1 and int(listFINALE[nHDFfiles+1][-24:-21])!=julianDayGranule\
            and int(strptime(str(int(datetime(dayInProcess.tm_year,dayInProcess.tm_mon,dayInProcess.tm_mday).strftime("%j"))+1),"%j").tm_mon)\
            == int(strptime(str(int(datetime(dayInProcess.tm_year,dayInProcess.tm_mon,dayInProcess.tm_mday).strftime("%j"))),"%j").tm_mon)):
            
            cond1 = 1.
        
        # "Normal" years
        if int(dayInProcess.tm_year)!=2008 and int(dayInProcess.tm_year)!=2012 and int(dayInProcess.tm_year)!=2016:
            """Month change :
            
            Before Oct 2017:
            if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                and int(listFINALE[nHDFfiles+1][-19:-16])!=julianDayGranule\
                and int(strptime(str(int(listFINALE[nHDFfiles+1][-19:-16])),"%j").tm_mon)\
                != int(strptime(str(int(listFINALE[nHDFfiles][-19:-16])),"%j").tm_mon)):
            
            Between Oct 2017 and Apr 2021 :
            if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                and int(listFINALE[nHDFfiles+1][-25:-22])!=julianDayGranule\
                and int(strptime(str(int(listFINALE[nHDFfiles+1][-25:-22])),"%j").tm_mon)\
                != int(strptime(str(int(listFINALE[nHDFfiles][-25:-22])),"%j").tm_mon)):
            """
            
            # Since Apr 2021 : 
            if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                and int(listFINALE[nHDFfiles+1][-24:-21])!=julianDayGranule\
                and int(strptime(str(int(listFINALE[nHDFfiles+1][-24:-21])),"%j").tm_mon)\
                != int(strptime(str(int(listFINALE[nHDFfiles][-24:-21])),"%j").tm_mon)):
                
                cond2 = 3.1
        
        # Leap years
        elif int(dayInProcess.tm_year)==2008 or int(dayInProcess.tm_year)==2012 or int(dayInProcess.tm_year)==2016:
            """
            Before Oct 2017:
            if int(listFINALE[nHDFfiles][-19:-16])>59:
                if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                and int(listFINALE[nHDFfiles+1][-19:-16])!=julianDayGranule\
                and  int(strptime(str(int(listFINALE[nHDFfiles+1][-19:-16])-1),"%j").tm_mon)\
                != int(strptime(str(int(listFINALE[nHDFfiles][-19:-16])-1),"%j").tm_mon)):                     
            
            Between Oct 2017 and Apr 2021 :
            if int(listFINALE[nHDFfiles][-25:-22])>59:
                if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                and int(listFINALE[nHDFfiles+1][-25:-22])!=julianDayGranule\
                and  int(strptime(str(int(listFINALE[nHDFfiles+1][-25:-22])-1),"%j").tm_mon)\
                != int(strptime(str(int(listFINALE[nHDFfiles][-25:-22])-1),"%j").tm_mon)):    
            """
            # Since Apr 2021 : 
            if int(listFINALE[nHDFfiles][-24:-21])>59:  # if it goes after the 29th of February
                if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                    and int(listFINALE[nHDFfiles+1][-24:-21])!=julianDayGranule\
                    and  int(strptime(str(int(listFINALE[nHDFfiles+1][-24:-21])-1),"%j").tm_mon)\
                    != int(strptime(str(int(listFINALE[nHDFfiles][-24:-21])-1),"%j").tm_mon)):     
                    
                    cond2 = 3.2
                    
            else:  # Before the 29th of February
                """
                Before Oct 2017:
                if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                    and int(listFINALE[nHDFfiles+1][-19:-16])!=julianDayGranule\
                    and int(strptime(str(int(listFINALE[nHDFfiles+1][-19:-16])),"%j").tm_mon)\
                    != int(strptime(str(int(listFINALE[nHDFfiles][-19:-16])),"%j").tm_mon)):                                          
                
                Between Oct 2017 and Apr 2021 :
                if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                    and int(listFINALE[nHDFfiles+1][-25:-22])!=julianDayGranule\
                    and int(strptime(str(int(listFINALE[nHDFfiles+1][-25:-22])),"%j").tm_mon)\
                    != int(strptime(str(int(listFINALE[nHDFfiles][-25:-22])),"%j").tm_mon)):
                """
                
                # Since Apr 2021 : 
                if (comptDay4segmentation!=0. and nHDFfiles!=len(listFINALE)-1\
                    and int(listFINALE[nHDFfiles+1][-24:-21])!=julianDayGranule\
                    and int(strptime(str(int(listFINALE[nHDFfiles+1][-24:-21])),"%j").tm_mon)\
                    != int(strptime(str(int(listFINALE[nHDFfiles][-24:-21])),"%j").tm_mon)):
                    
                    cond2 = 3.3
                
        # If it is the end of the list
        if nHDFfiles == len(listFINALE)-1:
            cond3 = 5.
        
        """STEP 8 - Writting in the text file
        
        """
        if cond1 + cond2 + cond3 > 0.:        
            writingOK = 0.
            nw = 0.
            goWriting = False
            
            if 'dardarPix_in_grid_Count' in locals():
                nw += 1.
                writingOK += 1.
            if 'pix_in_cloud_Count' in locals():
                nw += 1.
                writingOK += 1.
            if 'Allpix_in_cloud_Count' in locals():
                nw += 1.
                writingOK += 1.
            if 'cloud_occurence_in_pix_Count' in locals():
                nw += 1.
                writingOK += 1.                
            if 'Allpix_in_cloud_occur_Count' in locals():
                nw += 1.
                writingOK += 1.
            if IceAndSupercooled_mask == True:                 
                nw += 1.
                if 'iceAndSupercooled_Count' in locals(): writingOK += 1.
            if AllScooled_mask == True: 
                nw += 2.
                if 'allSupercooled_Count' in locals(): writingOK += 1.
                if 'allSupercooled2_Count' in locals(): writingOK += 1.
            if ScooledOnly_mask == True: 
                nw += 1.
                if 'supercooledOnly_Count' in locals(): writingOK += 1.
            if ScooledOnlyMPC_mask == True: 
                nw += 2.
                if 'supercooledOnlyMPC_Count' in locals(): writingOK += 1.
                if 'supercooledOnlyALONE_Count' in locals(): writingOK += 1.
            if IceOnly_mask == True: 
                nw += 1.
                if 'iceOnly_Count' in locals(): writingOK += 1.
            if IceContaining_cloud_1_mask == True: 
                nw += 1.
                if 'iceContaining_cloud_1_Count' in locals(): writingOK += 1.
            if IceContaining_cloud_2_mask == True: 
                nw += 1.
                if 'iceContaining_cloud_2_Count' in locals(): writingOK += 1.
            if liquidWarm_mask == True: 
                nw += 1.
                if 'liqWarm_Count' in locals(): writingOK += 1.
            if classif_mix_clouds == True:
                nw += 4.
                if 'mask_cloud_sc_ice_BelowAbove_Count' in locals(): writingOK += 1.
                if 'mask_cloud_sc_ice_BelowOnly_Count' in locals(): writingOK += 1.
                if 'mask_cloud_sc_ice_BelowAbove2_Count' in locals(): writingOK += 1.
                if 'mask_cloud_sc_ice_BelowOnly2_Count' in locals(): writingOK += 1.
                
            if temperature == True: 
                nw += 1.
                if 'tempe_sum' in locals(): writingOK += 1.
            if temperature_2m == True: 
                nw += 1.
                if 'tempe2m_sum' in locals(): writingOK += 1.
            if iwc_val == True:
                nw += 1.
                if 'iwc_sum' in locals(): writingOK += 1.
            if ext_val == True: 
                nw += 1.
                if 'ext_sum' in locals(): writingOK += 1.
            if re_val == True: 
                nw += 1.
                if 're_sum' in locals(): writingOK += 1.
            if lidarRatio_val == True: 
                nw += 1.
                if 'LR_sum' in locals(): writingOK += 1.
            
            if CloudSatPrecip == True :
                nw += 4
                if 'CSRain_count' in locals(): writingOK += 1.
                if 'CSSnow_count' in locals(): writingOK += 1.
                if 'CSMixed_count' in locals(): writingOK += 1.
                if 'CSAll_count' in locals(): writingOK += 1
            
            if CloudAnalysis == True:
                nw += 16
                if 'Cloud_Between_Lidar_count' in locals(): writingOK += 1.
                if 'MPC_count' in locals(): writingOK += 1.
                if 'MPCTemp_count' in locals(): writingOK += 1.
                if 'USLC_count' in locals(): writingOK += 1.
                if 'USLC2_count' in locals(): writingOK += 1.
                if 'USLCTemp_count' in locals(): writingOK += 1.
                if 'USLC2Temp_count' in locals(): writingOK += 1.
                if 'IC_count' in locals(): writingOK += 1.
                if 'ICTemp_count' in locals(): writingOK += 1.
                if 'WC_count' in locals(): writingOK += 1.
                if 'CC_count' in locals(): writingOK += 1.
                if 'CCMatrix_count' in locals(): writingOK += 1.

                if 'TotalClouds_count' in locals(): writingOK += 1.
                if 'TotalCloudsMatrix' in locals(): writingOK += 1.
                if 'SingleClouds_count' in locals(): writingOK += 1.
                if 'MultiClouds_count' in locals(): writingOK += 1.
            
                nw += 4
                if 'MPCMatrix_count' in locals(): writingOK += 1.
                if 'HeightMPC_count' in locals(): writingOK += 1.
                if 'TempMPC_count' in locals(): writingOK += 1.
                if 'ThicknessMPC_count' in locals(): writingOK += 1.
            
                nw += 4
                if 'USLCMatrix_count' in locals(): writingOK += 1.
                if 'HeightUSLC_count' in locals(): writingOK += 1.
                if 'TempUSLC_count' in locals(): writingOK += 1.
                if 'ThicknessUSLC_count' in locals(): writingOK += 1.
            
                nw += 4
                if 'USLC2Matrix_count' in locals(): writingOK += 1.
                if 'HeightUSLC2_count' in locals(): writingOK += 1.
                if 'TempUSLC2_count' in locals(): writingOK += 1.
                if 'ThicknessUSLC2_count' in locals(): writingOK += 1.
            
                nw += 4
                if 'ICMatrix_count' in locals(): writingOK += 1.
                if 'HeightIC_count' in locals(): writingOK += 1.
                if 'TempIC_count' in locals(): writingOK += 1.
                if 'ThicknessIC_count' in locals(): writingOK += 1.
            
                nw += 4
                if 'WCMatrix_count' in locals(): writingOK += 1.
                if 'HeightWC_count' in locals(): writingOK += 1.
                if 'TempWC_count' in locals(): writingOK += 1.
                if 'ThicknessWC_count' in locals(): writingOK += 1.
                
                nw += 8
                if 'Precip_count' in locals(): writingOK += 1.
                if 'PrecipMatrix_count' in locals(): writingOK += 1.
                if 'ColdRain_count' in locals(): writingOK += 1.
                if 'ColdRainMatrix_count' in locals(): writingOK += 1.
                if 'ColdPrecip_count' in locals(): writingOK += 1.
                if 'ColdPrecipMatrix_count' in locals(): writingOK += 1.
                if 'WarmPrecip_count' in locals(): writingOK += 1.
                if 'WarmPrecipMatrix_count' in locals(): writingOK += 1.
                
            if MeteoConditions == True:
                nw += 15
                if 'geopotential_700_count' in locals(): writingOK += 1.
                if 'spec_hum_700_count' in locals(): writingOK += 1.
                if 'temperature_700_count' in locals(): writingOK += 1.
                if 'geopotential_850_count' in locals(): writingOK += 1.
                if 'spec_hum_850_count' in locals(): writingOK += 1.
                if 'temperature_850_count' in locals(): writingOK += 1.
                if 'SST_Skin_count' in locals(): writingOK += 1.
                if 'surf_temp_count' in locals(): writingOK += 1.
                if 'surf_press_count' in locals(): writingOK += 1.
                
                if 'SH500m_count' in locals(): writingOK += 1.
                if 'SH1000m_count' in locals(): writingOK += 1.
                if 'SH1500m_count' in locals(): writingOK += 1.
                if 'pix500_count' in locals(): writingOK += 1.
                if 'pix1000_count' in locals(): writingOK += 1.
                if 'pix1500_count' in locals(): writingOK += 1.
            
            if Winds == True:
                nw += 6
                if 'u_winds_700_count' in locals(): writingOK += 1.
                if 'v_winds_700_count' in locals(): writingOK += 1.
                if 'u_winds_850_count' in locals(): writingOK += 1.
                if 'v_winds_850_count' in locals(): writingOK += 1.
                if 'Surface_Uwind_count' in locals(): writingOK += 1.
                if 'Surface_Vwind_count' in locals(): writingOK += 1.
            
            if Surface_Type == True: 
                nw += 3 
                if 'Surface_Land_count' in locals(): writingOK += 1.
                if 'Surface_Water_count' in locals(): writingOK += 1.
                if 'Surface_Ice_count' in locals(): writingOK += 1.
                
        
            if writingOK/nw == 1.:
                goWriting = True
            
            if goWriting==True and textFile==True:
                """
                Before Oct 2017:
                day2 = strptime(listFINALE[nHDFfiles][-23:-16],"%Y%j")
                Between Oct 2017 and Apr 2021 :
                day2 = strptime(listFINALE[nHDFfiles][-29:-22],"%Y%j")
                """
                # Since Apr 2021 :
                day2 = strptime(listFINALE[nHDFfiles][-28:-21],"%Y%j")
                print "................................................................."
                print "...........Text files creation, last file:", listFINALE[nHDFfiles]           

                # Header
                enteteStats =\
                "Latitude= " + str(latMin) + "\t to " + str(latMax) + "\n" +\
                "dlat= " + str(resol_lat_deg) + "\n" +\
                "Longitude= " + str(longMin) + " to " +str(longMax) + "\n" +\
                "dlong= " + str(resol_long_deg) + "\n" +\
                "Height(m)= " + str(altiMin) + " to " + str(altiMax) + "\n" +\
                "dz= 60 m. - nb pix Z = " + str(len(alti)) + "\n" +\
                "from " + str(day1.tm_mday)+"/"+str(day1.tm_mon)+"/"+str(day1.tm_year) +\
                " to " + str(day2.tm_mday)+"/"+str(day2.tm_mon)+"/"+str(day2.tm_year) + "\n"\
                "nDays\tnGranules\n" + str(comptDay4segmentation) + "\t" + str(nfiles) + "\n"
                
                # PixNum
                fw = open(pfig+"/"+filesFolder+"/PixNumGEO_COVER_"+vers+"_"+\
                    str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                    +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                fw.write("Dardar pixels number in each mesh grid:\n")
                fw.write(enteteStats)                                                                                                              
                f2 = writer(fw, delimiter='\t', lineterminator='\n')
                f2.writerows(dardarPix_in_grid_Count.T)
                fw.close()                
                # AllCloudyPix
                fw = open(pfig+"/"+filesFolder+"/AllCloudyPixGEO_COVER_"+vers+"_"+\
                    str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                    +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                fw.write("Cloudy pixels number in mesh grid:\n")                                                
                fw.write(enteteStats)              
                f2 = writer(fw, delimiter='\t', lineterminator='\n')
                f2.writerows(Allpix_in_cloud_Count.T)
                fw.close()
                # AllCloudOccur
                fw = open(pfig+"/"+filesFolder+"/AllCloudOccurPixGEO_COVER_"+vers+\
                    str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                    +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                fw.write("All cloudy pixels number in mesh grid:\n")                                                
                fw.write(enteteStats)              
                f2 = writer(fw, delimiter='\t', lineterminator='\n')
                f2.writerows(Allpix_in_cloud_occur_Count.T)
                fw.close()
                # CloudyPix
                fw = open(pfig+"/"+filesFolder+"/CloudyPixGEO_COVER_"+vers+"_"\
                    +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                    +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                fw.write("Cloudy pixels number in mesh grid:\n")
                fw.write(enteteStats)              
                f2 = writer(fw, delimiter='\t', lineterminator='\n')
                f2.writerows(pix_in_cloud_Count.T)
                fw.close()
                # CloudOccurPix
                fw = open(pfig+"/"+filesFolder+"/CloudOccurPixGEO_COVER_"+vers+"_"\
                    +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                    +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                fw.write("cloud occurence grid mesh pixels:\n")
                fw.write(enteteStats)
                f2 = writer(fw, delimiter='\t', lineterminator='\n')
                f2.writerows(cloud_occurence_in_pix_Count.T)
                fw.close()
                
                if IceAndSupercooled_mask==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/IceAndSC_GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice + SC:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(iceAndSupercooled_Count.T)
                    fw.close()
                    
                if AllScooled_mask==True and textFile==True:
                    # AllSC
                    fw = open(pfig+"/"+filesFolder+"/All_SCGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("All SC:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(allSupercooled_Count.T)
                    fw.close()
                    # AllSC2
                    fw = open(pfig+"/"+filesFolder+"/All_SC2GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("All SC 2:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(allSupercooled2_Count.T)
                    fw.close()
                    
                if ScooledOnly_mask==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/SCOnlyGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC Only:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(supercooledOnly_Count.T)
                    fw.close()
                    
                if ScooledOnlyMPC_mask==True and textFile==True:
                    # SCOnlyMPC
                    fw = open(pfig+"/"+filesFolder+"/SCOnlyMPCGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC Only MPC:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(supercooledOnlyMPC_Count.T)
                    fw.close()
                    # SCOnlyALONE
                    fw = open(pfig+"/"+filesFolder+"/SCOnlyALONEGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC Only ALONE:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(supercooledOnlyALONE_Count.T)
                    fw.close()   
                    
                if IceOnly_mask==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/IceOnlyGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice Only:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(iceOnly_Count.T)
                    fw.close()

                if IceContaining_cloud_1_mask==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/IceContainingCloud1_GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice Only:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(iceContaining_cloud_1_Count.T)
                    fw.close()

                if IceContaining_cloud_2_mask==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/IceContainingCloud2_GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice Only:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(iceContaining_cloud_2_Count.T)
                    fw.close()

                if liquidWarm_mask==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/LiquidWarmGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice Only:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(liqWarm_Count.T)
                    fw.close()
                    
                if classif_mix_clouds==True and textFile==True:
                    # SC_ice_belowAbove
                    fw = open(pfig+"/"+filesFolder+"/SC_ice_belowAboveGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC with ice above and below:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(mask_cloud_sc_ice_BelowAbove_Count.T)
                    fw.close()
                    # SC_ice_belowOnly
                    fw = open(pfig+"/"+filesFolder+"/SC_ice_belowOnlyGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC with ice below only:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(mask_cloud_sc_ice_BelowOnly_Count.T)
                    fw.close()
                    # SC_ice_belowAbove2
                    fw = open(pfig+"/"+filesFolder+"/SC_ice_belowAbove2GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC with ice above and below 2:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(mask_cloud_sc_ice_BelowAbove2_Count.T)
                    fw.close()
                    # SC_ice_belowOnly2
                    fw = open(pfig+"/"+filesFolder+"/SC_ice_belowOnly2GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SC with ice below only 2:\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(mask_cloud_sc_ice_BelowOnly2_Count.T)
                    fw.close()

                if temperature==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/TGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("T (sum):\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(tempe_sum.T)
                    fw.close()
                if temperature_2m==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/T2mGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("T2m (sum)):\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(tempe2m_sum.T)
                    fw.close()
                if iwc_val==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/IWCGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("IWC (IWC sum where pix is cold cloud):\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(iwc_sum.T)
                    fw.close()
                if ext_val==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/EXTGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("EXTINCTION (EXT sum where pix is cold cloud):\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ext_sum.T)
                    fw.close()
                if re_val==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/REGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("REFF (REFF sum where pix is cold cloud):\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(re_sum.T)
                    fw.close()
                if lidarRatio_val==True and textFile==True:
                    fw = open(pfig+"/"+filesFolder+"/LRGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("LR  (LR sum where pix is cold cloud):\n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(LR_sum.T)
                    fw.close()
                
                if CloudSatPrecip==True and textFile==True:
                    # CSRain
                    fw = open(pfig+"/"+filesFolder+"/CSRain_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("CSRain occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(CSRain_count.T)
                    fw.close()
                    # CSSnow
                    fw = open(pfig+"/"+filesFolder+"/CSSnow_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("CSSnow occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(CSSnow_count.T)
                    fw.close()
                    # CSMixed
                    fw = open(pfig+"/"+filesFolder+"/CSMixed_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("CSMixed occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(CSMixed_count.T)
                    fw.close()
                    # CSAll
                    fw = open(pfig+"/"+filesFolder+"/CSAll_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("CSAll occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(CSAll_count.T)
                    fw.close()
                
                if CloudAnalysis==True and textFile==True:
                    
                    # Cloud between lidar files
                    fw = open(pfig+"/"+filesFolder+"/CLOUDLIDAR_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum cloud between lidar extinction \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Cloud_Between_Lidar_count.T)
                    fw.close()
                    
                    # MPC files
                    fw = open(pfig+"/"+filesFolder+"/MPCGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("MPC occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(MPC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/MPCTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("MPC occurrence (sum), with Ttop<273K : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(MPCTemp_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/MPCMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 MPC (to be used for MPC properties) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(MPCMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/MPCTopHeight_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Height of the lowest MPC (sum), to be used with MPCMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(HeightMPC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/MPCTopTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Temperature of the lowest MPC (sum), to be used with MPCMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TempMPC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/MPCThickness_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Thickness of the lowest MPC (sum), to be used with MPCMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ThicknessMPC_count.T)
                    fw.close()
                    
                    # USLC/USLC2 files
                    fw = open(pfig+"/"+filesFolder+"/USLCGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("USLC occurrence (sum), SCW pixels only : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(USLC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLC2GEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("USLC2 occurrence (sum), could contain warm pixels : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(USLC2_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLCTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("USLC occurrence (sum), SCW pixels only and Ttop<273K : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(USLCTemp_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLC2Temp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("USLC2 occurrence (sum), could contain warm pixels and Ttop<273K : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(USLC2Temp_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLCMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 USLC (to be used for USLC properties) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(USLCMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLCTopHeight_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Height of the lowest USLC (sum), to be used with USLCMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(HeightUSLC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLCTopTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Temperature of the lowest USLC (sum), to be used with USLCMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TempUSLC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLCThickness_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Thickness of the lowest USLC (sum), to be used with USLCMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ThicknessUSLC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLC2Matrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 USLC2 (to be used for USLC2 properties) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(USLC2Matrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLC2TopHeight_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Height of the lowest USLC2 (sum), to be used with USLC2Matrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(HeightUSLC2_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLC2TopTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Temperature of the lowest USLC2 (sum), to be used with USLC2Matrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TempUSLC2_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/USLC2Thickness_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Thickness of the lowest USLC2 (sum), to be used with USLC2Matrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ThicknessUSLC2_count.T)
                    fw.close()
                    
                    # Ice cloud files
                    fw = open(pfig+"/"+filesFolder+"/IceCloudGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice cloud occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(IC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/IceCloudTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice cloud occurrence (sum), with Ttop<273K : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ICTemp_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/IceCloudMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 ice cloud (to be used for IceCloud properties) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ICMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/IceCloudTopHeight_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Height of the lowest ice cloud (sum), to be used with IceCloudMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(HeightIC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/IceCloudTopTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w") 
                    fw.write("Top Temperature of the lowest ice cloud (sum), to be used with IceCloudMatrix : \n")
                    fw.write(enteteStats) 
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TempIC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/IceCloudThickness_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w") 
                    fw.write("Thickness of the lowest ice cloud (sum), to be used with IceCloudMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ThicknessIC_count.T)
                    fw.close()
                    
                    # Warm cloud files
                    fw = open(pfig+"/"+filesFolder+"/WarmCloudGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Warm cloud occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(WC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/WarmCloudMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 warm cloud (to be used for WarmCloud properties) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(WCMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/WarmCloudTopHeight_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Height of the lowest WarmCloud (sum), to be used with WarmCloudMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(HeightWC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/WarmCloudTopTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Top Temperature of the lowest WarmCloud (sum), to be used with WarmCloudMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TempWC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/WarmCloudThickness_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Thickness of the lowest WarmCloud (sum), to be used with WarmCloudMatrix : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ThicknessWC_count.T)
                    fw.close()
                    
                    # Cold cloud files
                    fw = open(pfig+"/"+filesFolder+"/ColdCloudGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Cold cloud occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(CC_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/ColdCloudMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 cold cloud : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(CCMatrix_count.T)
                    fw.close()
                    
                    # Total cloud files
                    fw = open(pfig+"/"+filesFolder+"/TotalCloudsGEO_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Cloud occurrence (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TotalClouds_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/TotalCloudsMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 cloud : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(TotalCloudsMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SingleClouds_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing only 1 cloud : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(SingleClouds_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/MultiClouds_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 2 clouds : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(MultiClouds_count.T)
                    fw.close()
                    
                    # Precipitations
                    fw = open(pfig+"/"+filesFolder+"/Precip_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Layers of precipitations (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Precip_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/PrecipMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 layer of precipitations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(PrecipMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/ColdRain_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Layers of ColdRain (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ColdRain_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/ColdRainMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 layer of precipitations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ColdRainMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/ColdPrecip_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Layers of  Cold precipitations (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ColdPrecip_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/ColdPrecipMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 layer of Cold precipitations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(ColdPrecipMatrix_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/WarmPrecip_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Layers of Warm precipitations (sum) : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(WarmPrecip_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/WarmPrecipMatrix_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Sum of columns containing at least 1 layer of Warm precipitations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(WarmPrecipMatrix_count.T)
                    fw.close()
                
                if MeteoConditions==True and textFile==True:
                    
                    fw = open(pfig+"/"+filesFolder+"/GeopotentialHeight_700mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("700mb geopotential height at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(geopotential_700_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SpecificHumidity_700mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Specific humidity at 700mb and at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(spec_hum_700_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Temperature_700mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Temperature at 700mb and at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(temperature_700_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/GeopotentialHeight_850mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("850mb geopotential height at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(geopotential_850_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SpecificHumidity_850mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Specific humidity at 850mb and at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(spec_hum_850_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Temperature_850mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Temperature at 850mb and at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(temperature_850_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SST_Skin_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SST at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(SST_Skin_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SurfaceTemp_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Temperature(2m) at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(surf_temp_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SurfacePressure_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Surface pressure at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(surf_press_count.T)
                    fw.close()
                    
                    #Specific humidity (Not tested version)
                    fw = open(pfig+"/"+filesFolder+"/SH500m_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SH500m at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(SH500m_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/pix500_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Number of pixels to calculate the 0-500m-averaged SH : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(pix500_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SH1000m_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SH1000m at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(SH1000m_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/pix1000_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Number of pixels to calculate the 0-1000m-averaged SH : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(pix1000_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/SH1500m_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("SH1500m at cloud detection locations : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(SH1500m_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/pix1500_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Number of pixels to calculate the 0-1500m-averaged SH \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(pix1500_count.T)
                    fw.close()
                    
                if Winds==True and textFile==True:
                    
                    fw = open(pfig+"/"+filesFolder+"/Uwinds_700mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Eastern wind velocity at 700mb geopotential height : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(u_winds_700_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Vwinds_700mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Northern wind velocity at 700mb geopotential height : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(v_winds_700_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Uwinds_850mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Eastern wind velocity at 850mb geopotential height : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(u_winds_850_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Vwinds_850mb_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Northern wind velocity at 850mb geopotential height : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(v_winds_850_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Uwinds_Surface_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Eastern wind velocity at the surface : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Surface_Uwind_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Vwinds_Surface_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Northern wind velocity at the surface : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Surface_Vwind_count.T)
                    fw.close()
                
                if Surface_Type ==True and textFile==True:
                    
                    fw = open(pfig+"/"+filesFolder+"/Surface_Land_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Land Surface Cover : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Surface_Land_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Surface_Water_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Water Surface Cover : \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Surface_Water_count.T)
                    fw.close()
                    
                    fw = open(pfig+"/"+filesFolder+"/Surface_Ice_COVER_"+vers+"_"\
                        +str(day1.tm_year*10000+day1.tm_mon*100+day1.tm_mday)+"_to_"\
                        +str(day2.tm_year*10000+day2.tm_mon*100+day2.tm_mday)+".txt", "w")
                    fw.write("Ice Surface Cover: \n")
                    fw.write(enteteStats)
                    f2 = writer(fw, delimiter='\t', lineterminator='\n')
                    f2.writerows(Surface_Ice_count.T)
                    fw.close()

                """STEP 9 (Last) : Reseting the counters
                
                """
                comptDay4segmentation = 1. 
                nfiles = 0.
                
                del pix_in_cloud_Count, Allpix_in_cloud_Count, Allpix_in_cloud_occur_Count,\
                    cloud_occurence_in_pix_Count, dardarPix_in_grid_Count
                
                if IceAndSupercooled_mask == True:
                    del iceAndSupercooled_Count
                if AllScooled_mask == True: 
                    del allSupercooled_Count, allSupercooled2_Count
                if ScooledOnly_mask == True: 
                    del supercooledOnly_Count
                if ScooledOnlyMPC_mask == True: 
                    del supercooledOnlyMPC_Count, supercooledOnlyALONE_Count
                if IceOnly_mask == True:
                    del iceOnly_Count
                if IceContaining_cloud_1_mask == True:
                    del iceContaining_cloud_1_Count
                if IceContaining_cloud_2_mask == True:
                    del iceContaining_cloud_2_Count	
                if liquidWarm_mask == True:
                    del liqWarm_Count
                if classif_mix_clouds==True:
                    del mask_cloud_sc_ice_BelowAbove_Count, mask_cloud_sc_ice_BelowOnly_Count,\
                        mask_cloud_sc_ice_BelowAbove2_Count, mask_cloud_sc_ice_BelowOnly2_Count
                
                if temperature == True:
                    del tempe_sum
                if temperature_2m == True:
                    del tempe2m_sum
                if iwc_val == True:
                    del iwc_sum
                if ext_val == True:
                    del ext_sum
                if re_val == True:
                    del re_sum
                if lidarRatio_val == True:
                    del LR_sum
                    
                if CloudSatPrecip == True :
                    del CSRain_count, CSSnow_count, CSAll_count, CSMixed_count
                
                if CloudAnalysis == True:
                    del Cloud_Between_Lidar_count, MPC_count, MPCTemp_count, MPCMatrix_count,\
                        HeightMPC_count, TempMPC_count, ThicknessMPC_count,\
                        USLC_count, USLCTemp_count, USLCMatrix_count,\
                        HeightUSLC_count, TempUSLC_count, ThicknessUSLC_count,\
                        USLC2_count, USLC2Temp_count, USLC2Matrix_count,\
                        HeightUSLC2_count, TempUSLC2_count, ThicknessUSLC2_count,\
                        IC_count, ICTemp_count, ICMatrix_count,\
                        HeightIC_count, TempIC_count, ThicknessIC_count,\
                        WC_count, WCMatrix_count,\
                        HeightWC_count, TempWC_count, ThicknessWC_count,\
                        CC_count, CCMatrix_count,\
                        TotalClouds_count, TotalCloudsMatrix_count,\
                        SingleClouds_count, MultiClouds_count,\
                        Precip_count, PrecipMatrix_count,\
                        ColdRain_count, ColdRainMatrix_count,\
                        ColdPrecip_count, ColdPrecipMatrix_count,\
                        WarmPrecip_count, WarmPrecipMatrix_count
                    
                if MeteoConditions == True:
                    del geopotential_700_count, spec_hum_700_count, temperature_700_count,\
                        geopotential_850_count, spec_hum_850_count, temperature_850_count,\
                        SST_Skin_count, surf_temp_count, surf_press_count,\
                        SH500m_count, SH1000m_count, SH1500m_count, pix500_count, pix1000_count, pix1500_count
                    
                if Winds == True:
                    del u_winds_700_count, v_winds_700_count,\
                    u_winds_850_count, v_winds_850_count, Surface_Uwind_count, Surface_Vwind_count
                
                if Surface_Type == True: 
                    del Surface_Land_count, Surface_Water_count, Surface_Ice_count
                
                print "writing files done."
                
        print "...........next granule"
        
    # Ending prints
    print "Climato DARDAR V2 - CLOUD COVER STEREOGRAPHIC PROJECTION ON NORTH OCEAN (70-82N)"
    print "Parameters:"
    print "year(s):\t\t", lYears
    print "month(s):\t\t", m
    print "day(s):\t\t\t", d
    print "latitude limits:\t", int(latMin), "\tto ", int(latMax), "\t\tdegrees"
    print "longitude limits:\t", int(longMin), "\tto ", int(longMax), "\t\tdegrees"                                                                                                                                                                                                                                 
    print "horizontal resolution (latitude) and number of bins:", resol_lat_deg, nbLat    
    print "altitude limits:\t", int(altiMin), "\tto ", int(altiMax), "\tm"
    print "flag instru:", str(instruFlag), "  (1:lidar only; 2:radar only; 3: radarANDlidar; None:radarORlidar)"
    print "flag land/water:", landWaterFlag
    print "END."


"""MAIN PROGRAM
"""
# Parameters file
from AD_paramFile_GeoProj_COVER_SO_v12\
import local_fPath, local_fig_path, remote_fPath, filesFolder, remote_fig_path,\
annee, mois, jours, legClim, versFig,\
altiMin, altiMax, latMin, latMax, resol_lat_deg, longMin, longMax, resol_long_deg,\
segFile, instruFlag, landWaterFlag

# Local calculation
dardarCloud_geo_projection_v2(local_fPath, local_fig_path, filesFolder,\
annee, mois, jours, latMin, latMax, resol_lat_deg, longMin, longMax, resol_long_deg,\
segFile, altiMin, altiMax, instruFlag, landWaterFlag, dayNightFlag="False",\
ScooledOnly_mask=False, ScooledOnlyMPC_mask=False, IceAndSupercooled_mask=False,\
AllScooled_mask=False, IceOnly_mask=False, IceContaining_cloud_1_mask=False, 
IceContaining_cloud_2_mask=False, liquidWarm_mask=False, classif_mix_clouds=False,\
iwc_val=False, ext_val=False, re_val=False, lidarRatio_val=False, temperature=True, temperature_2m=True,\
CloudAnalysis=True, MeteoConditions=True, Winds=False, CloudSatPrecip=False, vers=versFig, Surface_Type = False, textFile=True)

""" Remote calculation (ICARE) (Not tested with CloudSatPrecip products)
dardarCloud_geo_projection_v2(remote_fPath, remote_fig_path, filesFolder,\
annee, mois, jours, latMin, latMax, resol_lat_deg, longMin, longMax, resol_long_deg,\
segFile, altiMin, altiMax, instruFlag, landWaterFlag, dayNightFlag= "False",\
ScooledOnly_mask=False, ScooledOnlyMPC_mask=False, IceAndSupercooled_mask=False,\
AllScooled_mask=False, IceOnly_mask=False, IceContaining_cloud_1_mask=False,\
IceContaining_cloud_2_mask=False, liquidWarm_mask=False, classif_mix_clouds=False,\
iwc_val=False, ext_val=False, re_val=False, lidarRatio_val=False, temperature=True, temperature_2m=True,\
CloudAnalysis=True, MeteoConditions=True, Winds=True, CloudSatPrecip=False, vers=versFig, Surface_Type = False, textFile=True)
"""