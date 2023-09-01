'''Definition of function that return an array of zones based on the lat
and lon of an array given. There are in total 6 zones
'''
#Definition of limits for the different zones in london

def zoneassig(lat, lon):
    # South East limits
    se_limit_lat = 51.523811 # everthing lower than this value is south east
    se_limit_lon = -0.101342 # everything greater than this value is south east

    # East limits
    e_limit_lat = 51.523811 # everything greater than this value is east
    e_limit_lon = -0.022306 # everything greater than this is east

    # South West limits
    sw_limit_lat = 51.501631 # everthing lower than this value is south east
    sw_limit_lon = -0.101342 # everything lower than this value is south east

    # West limits
    w_limit_lat = 51.501072  # everthing greater than this value is south east
    w_limit_lon = -0.182406  # everthing lower than this value is south east

    # Central limits
    c_toplimit_lat = 51.501072 # everthing greater than this value is central
    c_lowlimit_lat = 51.523811 # everything lower than this value is central

    c_toplimit_lon = -0.101342 # everything lower than this value is central
    c_lowlimit_lon = -0.182406 # everthing greater than this value is central

    # North limits
    n_toplimit_lon = -0.022306 # everything lower than this value is central
    n_lowlimit_lon = -0.182406 # everthing greater than this value is central

    n_lowlimit_lat = 51.523811 # everthing greater than this value is central

    #Definition of variable to return
    loc = []


    for station in lat.index:
        if lat[station] > c_toplimit_lat and lat[station]< c_lowlimit_lat and lon[station]< c_toplimit_lon and lon[station]>c_lowlimit_lon:
            loc.append('Central')
        elif lon[station] < n_toplimit_lon and lon[station] > n_lowlimit_lon and lat[station] > n_lowlimit_lat:
            loc.append('North')
        elif lon[station] < w_limit_lon and lat[station] > w_limit_lat:
            loc.append('West')
        elif lon[station] < sw_limit_lon and lat[station] < sw_limit_lat:
            loc.append('South_West')
        elif lon[station] > e_limit_lon and lat[station] > e_limit_lat:
            loc.append('East')
        elif lon[station] > se_limit_lon and lat[station] < se_limit_lat:
            loc.append('East')
        else:
            loc.append('Other')

    return loc
