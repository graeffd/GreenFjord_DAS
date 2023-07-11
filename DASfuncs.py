import numpy as np
import scipy.signal
import scipy.interpolate
import datetime
import os
import sys
import glob
import h5py

def sintela_to_datetime(sintela_times):
    '''returns an array of datetime.datetime ''' 
    days1970 = datetime.date(1970, 1, 1).toordinal()
    # Vectorize everything
    converttime = np.vectorize(datetime.datetime.fromordinal)
    addday_lambda = lambda x : datetime.timedelta(days=x)
    adddays = np.vectorize(addday_lambda)
    day = days1970 + sintela_times/1e6/60/60/24
    thisDateTime = converttime(np.floor(day).astype(int))
    dayFraction = day-np.floor(day)
    thisDateTime = thisDateTime + adddays(dayFraction)
    return thisDateTime

def get_Onyx_file_time(filepath):
    '''returns datetime.datetime of filename'''
    filename = os.path.basename(filepath)
    date = filename.split('_')[1]
    time = filename.split('_')[2]
    time = datetime.datetime.strptime('{}T{}'.format(date, time), '%Y-%m-%dT%H.%M.%S')
    return time

def get_Onyx_h5(dir_path, t_start, t_end=None, length=60):
    '''get list of availlable files within time range'''
    if not t_end:
        t_end = t_start + datetime.timedelta(seconds=length)
    files = []
    for file in glob.iglob(os.path.join(dir_path,'*')):
        file_timestamp = get_Onyx_file_time(file)
        if (file_timestamp>=t_start) & (file_timestamp<=t_end):
            files.append(file)
        else:
            pass
    files.sort()
    return files

# read multiple files
def read_Onyx_h5(files, cha_start=0, cha_end=-1):
    '''reads lists of hdf5 files saved by the Sintela Onyx DAS interrogator'''
    for i,file in enumerate(files):
        print("File {} of {}".format(i+1, len(files)), end="\r")
        try:
            f = h5py.File(file,'r')
            times_read = np.array(f['Acquisition/Raw[0]/RawDataTime'])
            data_read = np.array(f['Acquisition/Raw[0]/RawData'])[:, cha_start:cha_end]
            attrs_read = dict(f['Acquisition'].attrs)
            settings_read = (np.round(attrs_read['GaugeLength'],2), 
                            attrs_read['PulseRate'], 
                            attrs_read['PulseWidth'], 
                            np.round(attrs_read['SpatialSamplingInterval'],2), 
                            attrs_read['StartLocusIndex'], data_read.shape[1])


            if 'settings' not in locals():
                data = data_read
                times = times_read
                attrs = attrs_read
                settings = settings_read
            else:
                # check if basic settings are identical between files
                if settings_read == settings:
                    data = np.concatenate([data, data_read], axis=0)
                    times = np.concatenate([times, times_read], axis=0)
                else:
                    print('Aquisition parameters differ between files. No data returned')
                    data = np.array([])
                    times = np.array([])
                    attrs = dict()
                    return data, times, attrs
        except Exception as e:
            print('Problems with: {}'.format(file))
            print(e)
            data = np.array([])
            times = np.array([])
            attrs = dict()
            return data, times, attrs
    return data, times, attrs

def split_continuous_data(t_rec, data_rec, attrs):
    '''fills continuous (without data gaps) data into list'''
    data_list = []
    time_list = []

    # find indices where time difference is larger than PulseRate and split there
    dt = np.diff(t_rec)/1e6
    gap_idx = np.where(dt>1/attrs['PulseRate'])[0]
    
    if len(gap_idx)>1: # if data gaps present
        time_list.append(t_rec[:gap_idx[0]]) # first gapless data piece
        data_list.append(data_rec[:gap_idx[0]]) 
        for i in range(0,len(gap_idx)-1):
            time_list.append(t_rec[gap_idx[i]+1:gap_idx[i+1]+1]) # gapless data in between, +1 because of np.diff index shifted
            data_list.append(data_rec[gap_idx[i]+1:gap_idx[i+1]+1]) 
        data_list.append(data_rec[gap_idx[-1]+1:]) # last gapless data piece
        time_list.append(t_rec[gap_idx[-1]+1:])

        time_list = [i for i in time_list if len(i)>1] # remove empty list entries
        data_list = [i for i in data_list if len(i)>1] 
    else: # if no data gaps present
        data_list = [data_rec]
        time_list = [t_rec]
    return time_list, data_list

def fill_data_gaps(time_list, data_list, attrs, fill_value=np.nan, t_format=None):
    '''convert lists of gapless arrays into one array with fill_value at gaps'''
    # if len(time_list)>1:
    dt_eq = 1/attrs['PulseRate']*1e6
    t_eq = np.arange(time_list[0][0], time_list[-1][-1]+dt_eq, dt_eq) # equally sampled time array

    arr_filled = np.full((len(t_eq),data_list[0].shape[1]), fill_value) # array where to write the data to
    i=0 # index of arr_new
    for t_arr, d_arr in zip(time_list, data_list):
        while t_arr[0] > t_eq[i]: # fill with data only if recorded time is in equally sampled array (to within accuracy)
            i+=1
        arr_filled[i:i+len(t_arr)] = d_arr
        i+=len(t_arr)

    if t_format=='datetime':
        t_eq = sintela_to_datetime(t_eq)
    # else:
        
    return t_eq, arr_filled

def get_gaps(time_list, attrs, t_format=None):
    '''creates list with start and end times of data gaps'''
    gap_list = []
    dt_eq = 1/attrs['PulseRate']*1e6
    for i in range(len(time_list)-1):
        gap_list.append((time_list[i][-1]+dt_eq, time_list[i+1][0]-dt_eq)) # times of gaps
    if t_format=='datetime':
        gap_list = [(sintela_to_datetime(s), sintela_to_datetime(e)) for (s,e) in gap_list]
    return gap_list

def apply_sosfiltfilt_with_nan(sos, data, axis=-1, padtype='odd', padlen=None):
    '''applies the scipy.sosfiltfilt function and ignores errors'''
    try:
        filtered_data = scipy.signal.sosfiltfilt(sos, data, axis=axis, padtype=padtype, padlen=padlen)
        return filtered_data
    except Exception as e:
        warning_str = 'Returning an array filled with np.nan equally sized to the input array.'
        print('{}, {}'.format(e, warning_str), end='\r')
        return np.full(data.shape, np.nan)
    
def decimate(time_list, data_list, factor, attrs):
    '''decimates list of gapless arrays in time by factor and outputs one array with filled gaps'''
    sos = scipy.signal.butter(2, attrs['PulseRate']/factor/2.,'lowpass', fs=attrs['PulseRate'], output='sos') # frequency in m
    filt_list = [apply_sosfiltfilt_with_nan(sos, arr, axis=0) for arr in data_list]
    # fill filtered data into array
    t_cont, data_filt = fill_data_gaps(time_list, filt_list, attrs, t_format='datetime')
    data_dec = data_filt[::factor,:]
    t_dec = t_cont[::factor]
    return t_dec, data_dec
    
def interp_gaps(times_filled, data_filled, max_gap, **kwargs):
    '''interpolates NaNs in data up to a maximum data gap length in samples'''
    # select data gaps with a certain length
    gap_idxs = np.where(np.isnan(data_filled).all(axis=1))[0] # find all NaNs
    gaps_selected = []
    gap = []
    for i in range(len(gap_idxs)-1): # keep gap if shorter than max_gap
        gap.append(gap_idxs[i])
        if (gap_idxs[i+1]-gap_idxs[i] > 1):
            if len(gap)<max_gap:
                gaps_selected.extend(gap)
            gap = []

    # create boolean array from data gap indices
    gaps_bool = np.full(len(times_filled), False)
    gaps_bool[gaps_selected] = True

    # interpolate selected data gaps
    f_interp = scipy.interpolate.interp1d(times_filled[~gaps_bool], data_filled[~gaps_bool,:], axis=0)
    data_interp = data_filled.copy()
    data_interp[gaps_bool,:] = f_interp(times_filled[gaps_bool])
    
    return data_interp

def split_at_datagaps(times_filled, data_filled):
    '''splits arrays with NaNs into a list of contiguous arrays'''
    mask1d = np.isnan(data_filled).all(axis=1)
    new_data_list = [data_filled[s,:] for s in np.ma.clump_unmasked(np.ma.array(times_filled, mask=mask1d))]
    new_time_list = [times_filled[s] for s in np.ma.clump_unmasked(np.ma.array(times_filled, mask=mask1d))]
    return new_time_list, new_data_list






##### RECYCLING ######

# this might be deleted    
def _fill_data_gaps_with_nans(f):
    # fill data gaps with NaN values
    data = np.array(f['Acquisition/Raw[0]/RawData']) # read this in again to avoid malefunction if cell is run again
    t_rec = np.array(f['Acquisition/Raw[0]/RawDataTime'])
    dt_eq = 1/attrs['PulseRate']*1e6

    t_eq = np.arange(t_rec[0], t_rec[-1]+dt_eq, dt_eq) # equally sampled time array
    t_new = np.full(len(t_eq), np.nan) # same length array filled with NaN
    arr_new = np.full((len(t_eq),data.shape[1]), np.nan) # array where to write the data to

    i=0 # index of arr_new and t_new
    for t_rec_idx in range(len(t_rec)):   
        while t_rec[t_rec_idx] > t_eq[i]: # fill with data only if recorded time is in equally sampled array (to within accuracy)
            i+=1
        t_new[i] = t_rec[t_rec_idx]
        arr_new[i] = data[t_rec_idx,:]
    data = arr_new # override the data array
    times = sintela_to_datetime(t_eq)
    return

