"""
Originally created by G. Mioche, on 15 oct. 2012
Adapted and revised by C. Bazantay, on 30 apr. 2021
"""

"""PARAMETER FILE

remote_fPath : str
    file path on ICARE archive
remote_fig_path : str
    local figure folder path
    to define what using remote calculation
local_fPath : str
    file path on local storage
local_fig_path : str
    local figure folder path
annee : list of str
    list of years to process
    if annee = [], all available years will be processed
mois : list of str
    list of months to process
    if mois = [], all available months for selected years will be processed
jours : list of str
    list of days to process
    if jours = [], all available days for selected years and months will be processed
legClim : str
    Not used.
    In Mioche version, it was used to legend figures
versFig :
    Not used.
    In Mioche version, it was used to change name of the output files
altiMin, altiMax : int, int
    Minimum and maximum altitude in meters.
latMin, latMax : int, int
    Minimum and maximum latitude in degrees.
longMin, longMax : int, int
    Minimum and maximum longitude in degrees.
resol_lat_deg, resol_long_deg : int, int
    Resolution of mesh grid in output files, in degrees.
"""

""" Older remote_fPath :
remote_fPath="/DATA/LIENS/MULTI_SENSOR/DARDAR_CLOUD.v2.1.1/"
remote_fPath="/DATA/LIENS/CLOUDSAT/DARDAR-MASK.v2.11/"
"""
remote_fPath="/DATA/CLOUDSAT/DARDAR-MASK.v2.23/"
remote_fig_path="/home/mioche"

local_fPath="Z:/icare/"
local_fig_path="C:/Users/dziduch/Documents/M1-M2/DARDAR"

annee   = ["2007","2008","2009"]
mois    = []
jours = []
legClim = "Arctic (70-82N) stereographic projection"
versFig = ""

altiMin = 500.
altiMax = 3000.

latMin = 70.
latMax = 82.
resol_lat_deg = 2.

longMin = -180.
longMax = 180.
resol_long_deg = 5.

segFile = 7. ## Day numbers in txt files
instruFlag = None  ## 1=lidar only; 2=radar only; 3=radar+lidar
landWaterFlag = None

filesFolder="TxtFiles_SOCP_V1"