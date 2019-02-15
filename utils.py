import os
from itertools import repeat
import numpy as np
import pandas as pd

SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']

TRAIN_COLUMNS = ["label", "weight"]

FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T",
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_S"]

ID_COLUMN = "id"

N_STATIONS = 4
FEATURES_PER_STATION = 6
N_FOI_FEATURES = N_STATIONS*FEATURES_PER_STATION
# The value to use for stations with missing hits
# when computing FOI features
EMPTY_FILLER = 1000

# Examples on working with the provided files in different ways

# hdf is all fine - but it requires unpickling the numpy arrays
# which is not guranteed
def load_train_hdf(path):
    return pd.concat([
        pd.read_hdf(os.path.join(path, "train_part_%i_v2.hdf" % i))
        for i in (1, 2)], axis=0, ignore_index=True)

def load_train_hdf_head(path, fraq_n=1000):
    return pd.concat([
        pd.read_hdf(os.path.join(path, "train_part_%i_v2.hdf" % i)).sample(n=fraq_n)
        for i in (1, 2)], axis=0, ignore_index=True).reset_index(drop=True)

def load_test_data(path):
    test = pd.read_hdf(os.path.join(path, "test_public_v2.hdf"), index_col=ID_COLUMN)
    return test

def load_private_data(path):
    private = pd.read_hdf(os.path.join(path, "test_private_v2_track_1.hdf"), index_col=ID_COLUMN)
    return private

def load_data_csv(path, feature_columns):
    train = pd.concat([
        pd.read_csv(os.path.join(path, "train_part_%i_v2.csv.gz" % i),
                    usecols= [ID_COLUMN] + feature_columns + TRAIN_COLUMNS,
                    index_col=ID_COLUMN)
        for i in (1, 2)], axis=0, ignore_index=True)
    test = pd.read_csv(os.path.join(path, "test_public_v2.csv.gz"),
                       usecols=[ID_COLUMN] + feature_columns, index_col=ID_COLUMN)
    return train, test


def parse_array(line, dtype=np.float32):
    return np.fromstring(line[1:-1], sep=" ", dtype=dtype)


def load_full_test_csv(path):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    test = pd.read_csv(os.path.join(path, "test_public_v2.csv.gz"),
                       index_col="id", converters=converters,
                       dtype=types,
                       usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
    return test


def find_closest_hit_per_station_loc(row):
    EMPTY_FILLER = np.nan
    result = np.empty(44, dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]
    closest_distance_station = result[24:28]
    normed_distance_per_stations = result[28:32]
    distance_from_origin = result[32:36]
    closest_dz = result[36:40]
    close_match_dist = result[40:44]
    total_distance = 0
    hits_count = 0
    for station in range(4):
        hits = (row["FOI_hits_S"] == station)
 
        if not hits.any():
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
            closest_distance_station[station] = EMPTY_FILLER
            normed_distance_per_stations[station] = EMPTY_FILLER
            distance_from_origin[station] = EMPTY_FILLER
            closest_dz[station] = EMPTY_FILLER
            close_match_dist[station] = EMPTY_FILLER
        else:
#             import pdb;pdb.set_trace()
            lextra = [15270, 16470, 17670, 18870]
            x_distances_2 = (row["Lextra_X[%i]" % station] - row["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - row["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            closest_hit = np.argmin(distances_2)
            closest_distance_station[station] = distances_2[closest_hit]
            distance_from_origin[station] = row["FOI_hits_Z"][hits][closest_hit]**2 + row["FOI_hits_X"][hits][closest_hit]**2 + row["FOI_hits_Y"][hits][closest_hit]**2
            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = row["FOI_hits_T"][hits][closest_hit]
            closest_z_per_station[station] = row["FOI_hits_Z"][hits][closest_hit]
            closest_dx_per_station[station] = row["FOI_hits_DX"][hits][closest_hit]
            closest_dy_per_station[station] = row["FOI_hits_DY"][hits][closest_hit]
            close_match_dist[station] = (closest_x_per_station[station] - row['MatchedHit_X[%i]' % station]) ** 2 + (closest_y_per_station[station] - row['MatchedHit_Y[%i]' % station]) ** 2
            closest_dz[station] = (closest_z_per_station[station] - lextra[station])**2
            normed_distance_per_stations[station] = x_distances_2[closest_hit] / closest_dx_per_station[station]**2 + \
                y_distances_2[closest_hit] / closest_dy_per_station[station] ** 2
            total_distance += normed_distance_per_stations[station]
            hits_count += 1
    if hits_count > 1:
        total_distance /= hits_count

    pflags = np.array([(row['P'] < 6000), (row['P'] >=  6000 and row['P'] <  10000 ), (row['P'] >= 10000), 
                       (row['P'] <= 3500), (row['P'] < 6000 and row['P'] > 3500)]).astype(np.float32)
    flags = np.diag([1,1,1,1])[row["FOI_hits_S"]].sum(0).clip(max = 1).astype(np.float32)
    ismuon1 = flags[0] * flags[1] * (row['P'] < 6000)
    ismuon2 = flags[0] * flags[1] * (flags[2] + flags[3]).clip(max=1) * (row['P'] >=  6000 and row['P'] <  10000 )
    ismuon3 = flags[0] * flags[1] * flags[2] * flags[3] * (row['P'] >= 10000)
    ismuon = (ismuon1 + ismuon2 + ismuon3).clip(max=1)
    isloose1 = flags[0] * flags[1] * (row['P'] <= 3500)
    isloose2 = ((flags[0] + flags[1] + flags[2]) >= 2) * (row['P'] < 6000 and row['P'] > 3500 )
    isloose3 = ((flags[0] + flags[1] + flags[2] + flags[3]) >= 3) * (row['P'] >= 10000 )
    isloose = (isloose1 + isloose2 + isloose3).clip(max = 1)
    tight_hits = [x for i,x in enumerate(row["FOI_hits_S"]) if row['MatchedHit_TYPE[%i]' % x] == 2]
    if tight_hits:
        tight_flags = np.diag([1,1,1,1])[tight_hits].sum(0).clip(max = 1)
        istight1 = tight_flags[0] * tight_flags[1] * (row['P'] < 6000)
        istight2 = tight_flags[0] * tight_flags[1] * (tight_flags[2] + tight_flags[3]).clip(max=1) * (row['P'] >=  6000 and row['P'] <  10000 )
        istight3 = tight_flags[0] * tight_flags[1] * tight_flags[2] * tight_flags[3] * (row['P'] >= 10000)
        istight = (istight1 + istight2 + istight3).clip(max=1)
    else:
        istight = 0
    result = np.concatenate((result,pflags, flags,np.array([ismuon,isloose,istight,hits_count,total_distance])))
    return result

def find_closest_hit_per_station(row):
    result = np.empty(N_FOI_FEATURES, dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]
    
    for station in range(4):
        hits = (row["FOI_hits_S"] == station)
        if not hits.any():
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
        else:
            x_distances_2 = (row["Lextra_X[%i]" % station] - row["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - row["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            closest_hit = np.argmin(distances_2)
            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = row["FOI_hits_T"][hits][closest_hit]
            closest_z_per_station[station] = row["FOI_hits_Z"][hits][closest_hit]
            closest_dx_per_station[station] = row["FOI_hits_DX"][hits][closest_hit]
            closest_dy_per_station[station] = row["FOI_hits_DY"][hits][closest_hit]
    return result


def add_features(df):
    df['angle'] = np.arcsin(df['PT']/df['P'])
    df['pseudorap'] = - np.log(np.tan(df['angle'] / 2))
    df['r'] = np.sqrt(df['angle'] ** 2 + df['pseudorap'] **2)
    df['PZ'] = np.sqrt(df['P']** 2 -  df['PT']**2)