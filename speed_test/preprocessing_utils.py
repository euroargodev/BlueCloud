import numpy as np
import xarray as xr
import dask
from dask_ml.decomposition import PCA as dask_pca
from dask_ml.preprocessing import StandardScaler as dask_scaler
from sklearn.decomposition import PCA as sk_pca
from sklearn.preprocessing import StandardScaler as sk_scaler



def read_dataset(path, multiple, backend):
    if multiple:
        ds_full = xr.open_mfdataset(path, parallel=True)
    else:
        if backend == 'sk':
            ds_full = xr.open_dataset(path)
        else:
            ds_full = xr.open_dataset(path, chunks={"longitude": 500})

    print("size full DS: " + str(ds_full.nbytes / 1073741824) + ' Go')
    return ds_full


def select_var(ds_full, var_name, multiple, backend, spacial_domain="glob"):
    if spacial_domain=="glob":
        ds = ds_full[[var_name]]
    else:
        ds = ds_full.where((ds_full['longitude']>spacial_domain['lon'][0]) 
                           & (ds_full['longitude']<spacial_domain['lon'][1])
                           & (ds_full['latitude']>spacial_domain['lat'][0])
                           & (ds_full['latitude']<spacial_domain['lat'][1]) 
                           & (ds_full['depth']<spacial_domain['depth']), drop=True)[[var_name]]
    print("size after selection of variable: " + str(ds.nbytes / 1073741824) + ' Go')
    if multiple and backend == 'sk':
        ds.load()
    if backend == 'dask':
            ds[var_name].data = ds[var_name].data.rechunk({0:-1, 3: 100})
    return ds


def filter_profiles(x):
#     x = x.stack(sample_dim=('time', 'latitude', 'longitude'))
    x = x.stack(sample_dim=('longitude', 'latitude', 'time'))
    x = x.dropna(dim='sample_dim', how='any')
    return x


def count_NaN(x):
    return np.count_nonzero(np.isnan(x['thetao'].values))


def interpolation(x, dim_i, method='nearest', limit=10):
    # x[var_name_ds].interpolate_na(dim ='depth', method="nearest", limit=10, fill_value="extrapolate").to_dataset(name = var_name_ds)
    return x['thetao'].interpolate_na(dim=dim_i, method=method, limit=limit, fill_value="extrapolate").to_dataset(
        name='thetao')


def reformat_depth(x):
    x['depth'] = -np.abs(x['depth'].values)
    x.depth.attrs['axis'] = 'Z'
    return x


def drop_all_NaN(x):
    return x.dropna(dim='sample_dim', how='any')


def apply_pca(x, n_comp, var_name, backend):
    if backend == 'sk':
        pca = sk_pca(n_components=n_comp, svd_solver='full')
    else:
        pca = dask_pca(n_components=n_comp, svd_solver='full')
    pca.fit(x[var_name].data)
    x_reduced = pca.transform(x[var_name].data)
    x = x.assign(variables={var_name + "_reduced": (('sample_dim', 'feature_reduced'), x_reduced)})
    return x


def standard_scaling(x, backend, var_name):
    if backend == 'sk':
        scaler = sk_scaler()
    else:
        scaler = dask_scaler()
    scaler.fit(x[var_name].data)
    X_scale = scaler.transform(x[var_name].data)
    x = x.assign(variables={var_name + "_scaled": (('sample_dim', 'feature'), X_scale)})
    return x


def preprocessing_allin(path, scaling, backend, multiple, var_name, n_comp_pca, spacial_domain='glob'):
    ds = read_dataset(path, multiple, backend)
    ds = select_var(ds, var_name, multiple, backend, spacial_domain)
    x = filter_profiles(ds)
    #     interpolation and drop all nan not used since they are all filtered in "filter_profiles"
    #     x = interpolation(x, 'depth')
    #     x = drop_all_NaN(x)
    x = reformat_depth(x)
    x = x.transpose()
    if scaling:
        x = standard_scaling(x, backend, var_name)
        x = apply_pca(x=x, n_comp=n_comp_pca, var_name=var_name + '_scaled', backend=backend)
    else:
        x = apply_pca(x=x, n_comp=n_comp_pca, var_name=var_name, backend=backend)
    return x


def compute_quantile(x, var_name_ds, K, q):
    m_quantiles = x[var_name_ds].where(x['labels'] == 0, drop=True).chunk({'sample_dim': -1}).quantile(q, dim='sample_dim')
    for yi in range(1, K):
        m_quantiles = xr.concat((m_quantiles, x[var_name_ds].where(x['labels'] == yi, drop=True)
                                 .chunk({'sample_dim': -1}).quantile(q, dim='sample_dim')), dim='k')
    x = x.assign(variables={var_name_ds + "_Q": (('k', 'quantile', 'depth'), m_quantiles)})
    x = x.assign_coords(coords={'quantile': q})
    return x


def compute_quantile2(x, var_name_ds, K, q):
    def compute_one_q(array, k):
        return array[var_name_ds].where(array['labels']==k, drop=True).chunk({'sample_dim': -1}).quantile(q, dim='sample_dim')
    tmp = []
    for yi in range(K):
        a = dask.delayed(compute_one_q)(x, yi)
        tmp.append(a)
    m_quantiles = xr.concat(dask.compute(*tmp), dim='k')
    x = x.assign(variables={var_name_ds + "_Q":(('k','quantile','depth'), m_quantiles)})
    x = x.assign_coords(coords={'quantile': q, 'k': range(K)})
    return x