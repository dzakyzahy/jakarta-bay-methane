# -*- coding: utf-8 -*-
"""
plot_CH4_transek_v5.py
Interpolated CH4 / Flux map + transect line charts.
Non-interactive: runs both Water-CH4 and Flux variants automatically.

Data: 8apr/data/Gabungan _data_2023 dan 2024_dengan_nutrient_by_koordinat Dzaky.xlsx
      Sheet: Gabungan
Output: 8apr/outputfigure/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker
import xarray as xr
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from pykrige.uk import UniversalKriging
from adjustText import adjust_text

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  # 8apr/plot
APR8_DIR   = os.path.dirname(BASE_DIR)                   # 8apr
ROOT_DIR   = os.path.dirname(APR8_DIR)                   # uwaFajar
DATA_DIR   = os.path.join(APR8_DIR, 'data')              # 8apr/data
OUT_DIR    = os.path.join(APR8_DIR, 'outputfigure')      # 8apr/outputfigure

XLS_FILE   = os.path.join(DATA_DIR,
    'Gabungan _data_2023 dan 2024_dengan_nutrient_by_koordinat Dzaky.xlsx')
BATHY_FILE = os.path.join(ROOT_DIR, 'archive', 'telukjakarta_rev_6_bathy_fixed.nc')
COAST_SHP  = os.path.join(ROOT_DIR, 'coastlines-split-4326', 'lines.shp')

domain = [106.55, 107.05, -6.15, -5.84]  # lon_min, lon_max, lat_min, lat_max

# ── Colour map ─────────────────────────────────────────────────────────────────
colors_custom = ['#66CCFF', '#00CCFF', '#00FF00', '#FFFF00', '#FF6600', '#FF0000']
cmap_custom = LinearSegmentedColormap.from_list('custom_hot', colors_custom, N=256)
cmap_custom.set_over('#FF00FF')
cmap_custom.set_under('#66CCFF')

# ── Transect geometry ─────────────────────────────────────────────────────────
def calc_dist_km(lon1, lat1, lon2, lat2):
    mean_lat = (lat1 + lat2) / 2
    dx = (lon2 - lon1) * 111.0 * np.cos(np.radians(mean_lat))
    dy = (lat2 - lat1) * 111.0
    return np.sqrt(dx**2 + dy**2)


TARGET_LEN_KM = 10.0

dadap_start = (106.725079, -6.083330)
_orig_end   = (106.778436, -5.988163)
_ratio      = TARGET_LEN_KM / calc_dist_km(
    dadap_start[0], dadap_start[1], _orig_end[0], _orig_end[1])
dadap_end   = (
    dadap_start[0] + (_orig_end[0] - dadap_start[0]) * _ratio,
    dadap_start[1] + (_orig_end[1] - dadap_start[1]) * _ratio,
)

ancol_start   = (106.828444, -6.115589)
ancol_end     = (ancol_start[0], ancol_start[1] + TARGET_LEN_KM / 111.0)
marunda_start = (106.968451, -6.084779)
marunda_end   = (marunda_start[0], marunda_start[1] + TARGET_LEN_KM / 111.0)

TRANSECTS = [
    dict(name='Dadap',   start=dadap_start,   end=dadap_end,
         color='blue',  linestyle='-',  label='Dadap'),
    dict(name='Ancol',   start=ancol_start,   end=ancol_end,
         color='green', linestyle='--', label='Ancol'),
    dict(name='Marunda', start=marunda_start, end=marunda_end,
         color='red',   linestyle='-.', label='Marunda'),
]


# ══════════════════════════════════════════════════════════════════════════════
def load_raw():
    """Load all rows from Gabungan sheet; return raw DataFrame."""
    df = pd.read_excel(XLS_FILE, sheet_name='Gabungan', header=3)
    df.columns = df.columns.str.strip()
    df = df.replace('#N/A', np.nan)
    df = df.dropna(subset=['Lat', 'Lon'])
    return df


def derive_month(df):
    """
    Add a 'month_name' column derived from the Date column.
    Handles datetime objects and string dates.
    """
    import calendar

    def _to_month(v):
        if isinstance(v, (pd.Timestamp,)):
            return calendar.month_name[v.month].lower()
        if hasattr(v, 'month'):          # datetime.datetime
            return calendar.month_name[v.month].lower()
        if isinstance(v, str):
            for fmt in ['%d/%m/%Y', '%Y-%m-%d']:
                try:
                    import datetime
                    dt = datetime.datetime.strptime(v, fmt)
                    return calendar.month_name[dt.month].lower()
                except ValueError:
                    pass
        return 'unknown'

    df = df.copy()
    df['month_name'] = df['Date'].apply(_to_month)
    return df


def build_label_map(df):
    """
    Build a mapping from original station ID to simplified F/S labels.
    2023 stations → F1, F2, ...  (First)
    2024 stations → S1, S2, ...  (Second)
    Returns dict {original_id: (new_label, year)}.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['_year'] = df['Date'].dt.year

    label_map = {}
    for yr, prefix in [(2023, 'F'), (2024, 'S')]:
        yr_df = df[df['_year'] == yr]
        unique_ids = yr_df['ID Station'].astype(str).str.strip().unique()
        unique_ids = sorted(unique_ids)
        for i, sid in enumerate(unique_ids, start=1):
            label_map[sid] = (f'{prefix}{i}', yr)
    return label_map


def load_bathy():
    ds = xr.open_dataset(BATHY_FILE)
    lat_g  = ds['lat'].values
    lon_g  = ds['lon'].values
    bathy  = ds['bathy']
    b1 = np.array(bathy, dtype=float)
    b1[b1 > 0] = np.nan
    da_bathy = xr.DataArray(b1 * -1, coords=[lat_g, lon_g], dims=['lat', 'lon'])
    dum = np.array(bathy, dtype=float)
    dum[dum > 0] = np.nan
    dum_da   = xr.DataArray(dum * -1, coords=[lat_g, lon_g], dims=['lat', 'lon'])
    land_mask = (bathy >= 0) & (bathy.lon < 106.7) & (bathy.lat < -6.05)
    return da_bathy, dum, dum_da, lon_g, lat_g, bathy.shape, land_mask


def fetch_wms():
    try:
        from skimage import io as skio
        import requests
        from io import BytesIO
        url = ('https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?'
               'version=1.3.0&service=WMS&request=GetMap&format=image/jpeg'
               '&STYLE=default&bbox={},{},{},{}&CRS=EPSG:4326'
               '&HEIGHT=512&WIDTH=512&layers=BlueMarble_NextGeneration'
               ).format(domain[2], domain[0], domain[3], domain[1])
        resp = requests.get(url, verify=False, timeout=15)
        return skio.imread(BytesIO(resp.content))
    except Exception as e:
        print(f'  WMS not available: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
def run_variant(var_config, df_raw, da_bathy, dum, dum_da,
                lon_g, lat_g, bathy_shape, land_mask, img_wms, coastlines):
    """Run one full figure variant (Water CH4 or Flux)."""

    col        = var_config['col']
    units      = var_config['units']
    valmin     = var_config['valmin']
    valmax     = var_config['valmax']
    suffix     = var_config['suffix']
    prefix     = var_config.get('prefix', '')
    month_filt = var_config['month']        # 'august', 'november', or 'all'
    year_filt  = var_config.get('year', 'all')  # 2023, 2024, or 'all'

    # ── Prepare data ──────────────────────────────────────────────────────────
    df = df_raw.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Lat', 'Lon', col])
    df = df[df[col] > 0]                    # remove zeros

    # Filter by year
    if year_filt != 'all':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].dt.year == int(year_filt)]
        print(f'  → Year {year_filt} filter: {len(df)} points')

    # Filter by month
    if month_filt != 'all':
        df = df[df['month_name'].str.lower() == month_filt]
        print(f'  → {month_filt.capitalize()} filter: {len(df)} points')
    else:
        print(f'  → All months: {len(df)} points')

    if len(df) < 3:
        print(f'  ⚠ Not enough data points for interpolation, skipping {suffix}.')
        return

    lons    = df['Lon'].values
    lats    = df['Lat'].values
    vals    = df[col].values
    codes   = df['ID Station'].astype(str).str.strip().values
    months_ = df['month_name'].values

    # Build simplified labels: F=2023, S=2024
    label_map = build_label_map(df_raw)
    display_codes = []
    code_years    = []
    for c in codes:
        if c in label_map:
            lbl, yr = label_map[c]
            display_codes.append(lbl)
            code_years.append(yr)
        else:
            display_codes.append(c)
            code_years.append(None)
    display_codes = np.array(display_codes)
    code_years    = np.array(code_years)

    # ── Grid ──────────────────────────────────────────────────────────────────
    dim_y, dim_x = bathy_shape
    xmin, xmax = lon_g.min(), lon_g.max()
    ymin, ymax = lat_g.min(), lat_g.max()
    grid_lon = np.linspace(xmin, xmax, dim_x)
    grid_lat = np.linspace(ymin, ymax, dim_y)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    sparse_pts = np.column_stack((lons.ravel(), lats.ravel()))
    dense_pts  = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # ── Interpolation (RBF + weighted radius blending) ────────────────────────
    radius_deg    = 0.04
    rbf_smoothing = 0.2
    gaussian_sig  = 1.0

    print('  → RBF interpolation...')
    rbf = RBFInterpolator(sparse_pts, vals.ravel(),
                          smoothing=rbf_smoothing, kernel='cubic', epsilon=1)
    z_rbf = rbf(dense_pts).reshape(grid_x.shape)
    z_rbf = gaussian_filter(z_rbf, sigma=gaussian_sig)

    print('  → Weighted radius blending...')
    w_sum = np.zeros_like(grid_x, dtype=float)
    w_tot = np.zeros_like(grid_x, dtype=float)
    for lon0, lat0, val0 in zip(lons, lats, vals):
        dist = np.sqrt((grid_x - lon0)**2 + (grid_y - lat0)**2)
        in_r = dist < radius_deg
        w = 1 - dist[in_r] / radius_deg
        w_sum[in_r] += z_rbf[in_r] * w
        w_tot[in_r] += w

    with np.errstate(invalid='ignore', divide='ignore'):
        z_blend = w_sum / w_tot

    # Only show near data points; mask land
    mask_r = np.zeros_like(grid_x, dtype=bool)
    for lon0, lat0 in zip(lons, lats):
        dist = np.sqrt((grid_x - lon0)**2 + (grid_y - lat0)**2)
        mask_r |= dist < radius_deg

    z_blend = np.where(mask_r, z_blend, np.nan)
    final   = np.where(np.isnan(dum), np.nan, z_blend)

    masked_da = xr.DataArray(final,
                             coords={'lat': grid_lat, 'lon': grid_lon},
                             dims=['lat', 'lon'])
    masked_da = masked_da.where(~land_mask)

    # ── Transect sampling ─────────────────────────────────────────────────────
    n_pts = 1000
    transect_data = {}
    for t in TRANSECTS:
        lons_line = np.linspace(t['start'][0], t['end'][0], n_pts)
        lats_line = np.linspace(t['start'][1], t['end'][1], n_pts)
        dist_km = np.zeros(n_pts)
        for i in range(1, n_pts):
            dist_km[i] = dist_km[i-1] + calc_dist_km(
                lons_line[i-1], lats_line[i-1], lons_line[i], lats_line[i])
        xi = (lons_line - xmin) / (xmax - xmin) * (dim_x - 1)
        yi = (lats_line - ymin) / (ymax - ymin) * (dim_y - 1)
        sampled = map_coordinates(final, [yi, xi], order=1, mode='nearest')
        transect_data[t['name']] = dict(
            lons=lons_line, lats=lats_line, dist=dist_km, vals=sampled,
            color=t['color'], linestyle=t['linestyle'], label=t['label'])

    # ══ FIGURE 1: Map ══════════════════════════════════════════════════════════
    print('  → Building map figure...')
    fig1, ax1 = plt.subplots(1, 1, figsize=(19, 11),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    ax1.set_extent(domain, crs=ccrs.PlateCarree())

    if img_wms is not None:
        ax1.imshow(img_wms,
                   extent=(domain[0], domain[1], domain[2], domain[3]),
                   origin='upper', zorder=1)

    bathy_plot = da_bathy.plot(ax=ax1, x='lon', y='lat', cmap='Blues',
                               vmin=0, vmax=60, add_colorbar=False,
                               alpha=0.85, zorder=2)

    # Interpolated CH4
    plot_kw = dict(ax=ax1, x='lon', y='lat', cmap=cmap_custom,
                   add_colorbar=False, alpha=0.80,
                   vmin=valmin, vmax=valmax, zorder=3)
    c = masked_da.plot(**plot_kw)

    # Transect lines
    t_offsets = {'Dadap': -0.02, 'Ancol': -0.025, 'Marunda': -0.04}
    for t in TRANSECTS:
        ax1.plot([t['start'][0], t['end'][0]],
                 [t['start'][1], t['end'][1]],
                 color='black', linestyle=t['linestyle'],
                 linewidth=2, alpha=0.9, zorder=5)
        yoff = t_offsets.get(t['label'], -0.02)
        ax1.text(t['start'][0], t['start'][1] + yoff, t['label'],
                 color='black', fontsize=14, ha='center', va='top',
                 weight='bold',
                 bbox=dict(facecolor='white', alpha=0.95,
                           edgecolor='black', pad=2), zorder=30)

    # Scatter points coloured by month and year
    month_ser = pd.Series(months_).str.lower()
    legend_els = []
    
    plotted_months = set()
    for yr, fill_col in [(2023, '#1565C0'), (2024, '#2E7D32')]:
        for m_name, marker, ms in [('july',     'D', 80),
                                   ('august',   'o', 80),
                                   ('october',  's', 80),
                                   ('november', '*', 120)]:
            mask = (month_ser == m_name) & (code_years == yr)
            if mask.any():
                plotted_months.add(m_name)
                ax1.scatter(lons[mask], lats[mask],
                            facecolors=fill_col, edgecolors='white',
                            linewidth=1.0, s=ms, alpha=1, marker=marker, zorder=10)

    for m_name, marker in [('july', 'D'), ('august', 'o'), ('october', 's'), ('november', '*')]:
        if m_name in plotted_months:
            legend_els.append(mlines.Line2D(
                [], [], color='w', marker=marker, markerfacecolor='gray',
                markeredgecolor='white', markeredgewidth=1.0,
                markersize=10 if marker != '*' else 13,
                label=m_name.capitalize()))

    # Station labels — simplified: F=2023(blue), S=2024(green)
    YEAR_COLORS = {2023: '#1565C0', 2024: '#2E7D32'}  # blue, green
    texts, x_pt, y_pt = [], [], []
    for lon0, lat0, dcode, yr in zip(lons, lats, display_codes, code_years):
        if dcode.startswith('P'):
            continue
        lbl_color = YEAR_COLORS.get(yr, 'black')
        txt = ax1.text(lon0, lat0, dcode, fontsize=10, color=lbl_color,
                       weight='bold', zorder=15, ha='left', va='bottom',
                       path_effects=[
                           path_effects.withStroke(linewidth=2, foreground='white')
                       ])
        texts.append(txt)
        x_pt.append(lon0)
        y_pt.append(lat0)

    adjust_text(texts, x=x_pt, y=y_pt,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5,
                                shrinkA=10, shrinkB=10),
                expand_points=(5, 5), expand_text=(5, 5),
                force_points=(2.0, 2.0), force_text=(2.0, 2.0),
                only_move={'points': 'xy', 'text': 'xy'}, lim=3000)

    # Year prefix legend
    year_legend_els = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor='#1565C0',
                      markeredgecolor='white', label='F = 2023', markersize=10),
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor='#2E7D32',
                      markeredgecolor='white', label='S = 2024', markersize=10),
    ]
    all_legend = legend_els + year_legend_els
    if all_legend:
        ax1.legend(handles=all_legend, loc='upper right',
                   title='Sampling Month / Year', fontsize=11, title_fontsize=12)

    # Colorbars
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='3%', pad=0.08, axes_class=plt.Axes)
    cb1 = fig1.colorbar(c, cax=cax1, orientation='vertical', extend='neither')
    cb1.set_label(units, fontsize=18, fontweight='bold')
    cb1.ax.tick_params(labelsize=16)
    for lbl in cb1.ax.get_yticklabels():
        lbl.set_fontsize(16)
        lbl.set_fontweight('bold')

    cax2 = divider.append_axes('right', size='3%', pad=1.0, axes_class=plt.Axes)
    cb2 = fig1.colorbar(bathy_plot, cax=cax2, orientation='vertical', extend='neither')
    cb2.set_label('Depth (m)', fontsize=18, fontweight='bold')
    cb2.ax.tick_params(labelsize=16)
    for lbl in cb2.ax.get_yticklabels():
        lbl.set_fontsize(16)
        lbl.set_fontweight('bold')

    # Depth contours
    CL_5 = ax1.contour(grid_lon, grid_lat, dum_da, [5],
                       transform=ccrs.PlateCarree(), linewidths=1.5,
                       colors='black', linestyles='-', origin='lower', zorder=4)
    cl5_lbl = ax1.clabel(CL_5, inline_spacing=0.5, fontsize=18,
                         colors='black', fmt='%.0fm', zorder=4)
    for t in cl5_lbl:
        t.set_fontweight('bold')

    CL_d = ax1.contour(grid_lon, grid_lat, dum_da, [60, 70],
                       transform=ccrs.PlateCarree(), linewidths=1,
                       colors='white', linestyles='--', origin='lower', zorder=5)
    ax1.clabel(CL_d, inline_spacing=0.5, fontsize=13,
               colors='white', fmt='%.0fm', zorder=5)

    # Gridlines
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=1, color='gray', alpha=0.25, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(106.60, 107.1, 0.10))
    gl.xformatter = LongitudeFormatter(number_format='.2f', degree_symbol='',
                                       dateline_direction_label=True)
    gl.ylocator = mticker.FixedLocator(np.arange(-6.20, -5.80, 0.10))
    gl.yformatter = LatitudeFormatter(number_format='.2f', degree_symbol='')
    gl.xlabel_style = {'size': 20, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 20, 'color': 'black', 'weight': 'bold'}

    if coastlines is not None:
        coastlines.plot(color='orange', ax=ax1, zorder=4)

    # Inset: Pulau Pari
    pari_box = [106.5679, 106.643, -5.87, -5.845]
    ax1.add_patch(Rectangle((pari_box[0], pari_box[2]),
                             pari_box[1] - pari_box[0],
                             pari_box[3] - pari_box[2],
                             edgecolor='white', fill=False,
                             linestyle='-', lw=2, zorder=20))

    axins = inset_axes(ax1, width='40%', height='40%', loc='upper center',
                       bbox_to_anchor=(-0.05, 0.01, 1, 1),
                       bbox_transform=ax1.transAxes,
                       axes_class=cartopy.mpl.geoaxes.GeoAxes,
                       axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    axins.set_extent(pari_box, crs=ccrs.PlateCarree())

    if img_wms is not None:
        axins.imshow(img_wms,
                     extent=(domain[0], domain[1], domain[2], domain[3]),
                     origin='upper', zorder=1)

    da_bathy.plot(ax=axins, x='lon', y='lat', cmap='Blues', vmin=0, vmax=60,
                  add_colorbar=False, alpha=0.85, zorder=2)
    inset_kw = plot_kw.copy()
    inset_kw['ax'] = axins
    inset_kw['zorder'] = 3
    masked_da.where(~land_mask).plot(**inset_kw)

    if coastlines is not None:
        coastlines.plot(ax=axins, color='orange', linewidth=1, zorder=4)

    # P-points in inset
    p_df = pd.read_excel(XLS_FILE, sheet_name='Gabungan', header=3)
    p_df.columns = p_df.columns.str.strip()
    p_df = p_df.replace('#N/A', np.nan).dropna(subset=['Lat', 'Lon'])
    p_df = p_df[p_df['ID Station'].astype(str).str.startswith('P')]
    month_marker = {'july': 'D', 'august': 'o', 'october': 's', 'november': '*'}
    month_size = {'july': 80, 'august': 80, 'october': 80, 'november': 120}

    for _, row in p_df.iterrows():
        orig_id = str(row['ID Station'])
        mapped = label_map.get(orig_id)
        if mapped:
            new_id, yr = mapped
        else:
            new_id, yr = orig_id, 2023

        lbl_color = '#1565C0' if yr == 2023 else '#2E7D32'
        
        month_name = 'november'
        if pd.notnull(row.get('Date')):
            try:
                month_name = pd.to_datetime(row['Date']).strftime('%B').lower()
            except:
                pass
        
        marker = month_marker.get(month_name, '*')
        ms = month_size.get(month_name, 120)

        axins.scatter(row['Lon'], row['Lat'], marker=marker, s=ms,
                      facecolors=lbl_color, edgecolors='white',
                      linewidth=1.0, zorder=25)
        axins.annotate(new_id,
                       (row['Lon'], row['Lat']),
                       xytext=(4, 4), textcoords='offset points',
                       fontsize=10, color=lbl_color, weight='bold',
                       zorder=30, ha='left', va='bottom').set_path_effects(
            [path_effects.withStroke(linewidth=1.5, foreground='white')])

    ax1.add_artist(ConnectionPatch(
        xyA=(0, 0), coordsA=axins.transAxes,
        xyB=(pari_box[0], pari_box[2]), coordsB=ax1.transData,
        color='white', linewidth=2, linestyle='--', zorder=20))
    ax1.add_artist(ConnectionPatch(
        xyA=(1, 1), coordsA=axins.transAxes,
        xyB=(pari_box[1], pari_box[3]), coordsB=ax1.transData,
        color='white', linewidth=2, linestyle='--', zorder=20))

    ax1.text(106.70, -5.96, 'Jakarta Bay', fontsize=32, color='black', zorder=10)

    # ── Rivers (Tolong dirapihkan) ─────────────────────────────────────────────
    ds = 0.01
    
    #1
    ax1.scatter(106.96819, -6.09258, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.96819+ds*1, -6.09258-ds, 'BKT', fontsize=6, color='yellow', weight='bold', zorder=20)
    #2
    ax1.scatter(106.82769, -6.11542-ds, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.scatter(106.90539-ds*6, -6.09636, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.90539-6*ds, -6.09636-ds*3., 'Tj. Priok', fontsize=6, color='yellow', weight='bold', zorder=20)
    #3
    ax1.scatter(106.90539, -6.09636-ds, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.90539-ds, -6.09636-ds*3.0, 'Sunter', fontsize=6, color='yellow', weight='bold', zorder=20)
    #4
    ax1.scatter(106.93985, -6.09712, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.93985-ds*2, -6.09712-ds*1, 'Cilincing', fontsize=6, color='yellow', weight='bold', zorder=20)
    #5p
    ax1.scatter(106.95694, -6.09711, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    #6
    ax1.scatter(106.9939, -6.0253, color='white', marker='^', s=70, edgecolors='k', zorder=20)   # koordinat-nya salah
    ax1.text(106.9939-ds, -6.0253-ds*1.7, 'M. \nGembong', fontsize=6, color='yellow', weight='bold', zorder=20)
    #7
    ax1.scatter(106.79733, -6.10525, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.79733-ds*0.5, -6.10525-ds*2.58, 'Pompa \nPluit', fontsize=6, color='yellow', weight='bold', zorder=20)
    #8
    ax1.scatter(106.78519, -6.10185-ds, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.78519-ds*1.7, -6.10185-ds*2.7, 'Muara \nKarang', fontsize=6, color='yellow', weight='bold', zorder=20)
    #9
    ax1.scatter(106.76621, -6.09773-ds, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.76621-ds*1, -6.09773-ds*2.91, 'Muara \nAngke', fontsize=6, color='yellow', weight='bold', zorder=20)
    #10
    ax1.scatter(106.75238, -6.09819-ds, color='white', marker='^', s=70, edgecolors='k', zorder=20)   # salah koordinat
    ax1.text(106.75238-ds*1.5, -6.09819-ds*2.81, 'Cgk. \nDrain', fontsize=6, color='yellow', weight='bold', zorder=20)
    #11
    ax1.scatter(106.73461, -6.08078-ds*4, color='white', marker='^', s=70, edgecolors='k', zorder=20)
    ax1.text(106.73461-ds*2.5, -6.08078-ds*5, 'M.Kamal', fontsize=6, color='yellow', weight='bold', zorder=20)
    #12
    ax1.scatter(106.9888, -5.94115, color='white', marker='^', s=70, edgecolors='k', zorder=20) # diluar domain
    ax1.text(106.9888+ds*0.5, -5.94115, 'Citarum', fontsize=8, color='yellow', weight='bold', zorder=20)
    #13
    ax1.scatter(106.7228361, -6.08573889-ds*1, color='white', marker='^', s=20, edgecolors='k', zorder=20)
    ax1.scatter(106.9888, -5.94115, color='orange', marker='^', s=150, edgecolors='k', zorder=19) # simbol sungai muara dadap
    
    ax1.scatter(106.63833, -6.00057, color='white', marker='^', s=70, edgecolors='k', zorder=20) # diluar domain
    ax1.text(106.63833-ds*0.5, -6.00057-ds*1.5, 'Cisadane', fontsize=8, color='yellow', weight='bold', zorder=20)

    ax1.text(106.7264361+ds, -6.082838889-ds, 'PIK', fontsize=8, color='white', weight='bold', zorder=20)

    map_file = os.path.join(OUT_DIR, f'{prefix}Map_CH4_Transect_V5{suffix}.png')
    print(f'  → Saving map: {map_file}')
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ══ FIGURE 2: Line chart ═══════════════════════════════════════════════════
    print('  → Building transect line chart...')
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    x_common = np.linspace(0, TARGET_LEN_KM, 600)

    for name, tdata in transect_data.items():
        x_all = tdata['dist']
        y_all = tdata['vals']
        valid = ~np.isnan(y_all)
        xv, yv = x_all[valid], y_all[valid]

        if len(xv) < 5:
            print(f'    ⚠ {name}: insufficient data, skipping')
            continue
        try:
            fi = interp1d(xv, yv, kind='cubic', bounds_error=False,
                          fill_value=(yv[0], yv[-1]))
            xp = np.linspace(xv.min(), TARGET_LEN_KM, 500)
            yp = np.maximum(fi(xp), 0)
            yp = gaussian_filter1d(yp, sigma=3)
            yp = np.maximum(yp, 0)
            ax2.plot(xp, yp, label=tdata['label'],
                     color=tdata['color'], linestyle=tdata['linestyle'],
                     linewidth=2.5, alpha=0.9)
        except Exception as e:
            print(f'    ⚠ {name} smooth failed ({e}), plotting raw')
            ax2.plot(xv, yv, label=tdata['label'],
                     color=tdata['color'], linestyle=tdata['linestyle'],
                     linewidth=2.5, alpha=0.9)

    ax2.set_xlabel('Distance (km)', fontsize=20, fontweight='bold')
    ax2.set_ylabel(units, fontsize=20, fontweight='bold')
    month_label = month_filt.capitalize() if month_filt != 'all' else 'Jakarta Bay - All Months'
    ax2.set_title(f'{units} — Transects from River to Sea\n'
                  f'({month_label}, ref length: {TARGET_LEN_KM:.0f} km)',
                  fontsize=20, fontweight='bold', pad=12)
    ax2.set_xlim(0, TARGET_LEN_KM)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=17, loc='best', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax2.tick_params(axis='both', which='major', labelsize=17)

    graph_file = os.path.join(OUT_DIR, f'{prefix}Graph_CH4_Transect_V5{suffix}.png')
    print(f'  → Saving graph: {graph_file}')
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    run_mode = 'simple'
    if len(sys.argv) > 1 and sys.argv[1] in ['simple', 'lanjutan']:
        run_mode = sys.argv[1]

    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading data...')
    df_raw = load_raw()
    df_raw = derive_month(df_raw)
    print(f'  {len(df_raw)} stations loaded')
    print(f'  Month distribution:\n{df_raw["month_name"].value_counts().to_string()}')

    print('\nLoading bathymetry...')
    da_bathy, dum, dum_da, lon_g, lat_g, bathy_shape, land_mask = load_bathy()

    print('\nFetching WMS satellite image...')
    img_wms = fetch_wms()

    coastlines = None
    if os.path.exists(COAST_SHP):
        coastlines = gpd.read_file(COAST_SHP,
                                   bbox=(domain[0], domain[3], domain[1], domain[2]))

    # ── Define variants to produce ─────────────────────────────────────────────
    # DissCH4 is already in µmol/L — no conversion needed
    # Flux is already in µmol/m²/day  — no conversion needed
    variants = [
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane',
             month='all'),
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux',
             month='all'),
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_Aug',
             month='august'),
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_Nov',
             month='november'),
        # ── Per-year spatial maps (All months) ──────────────────────────────
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2023_All',
             month='all', year=2023),
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2024_All',
             month='all', year=2024),
        # ── Per-year per-month spatial maps ────────────────────────────────
        # 2023: July, August, October
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2023_Jul',
             month='july', year=2023),
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2023_Aug',
             month='august', year=2023),
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2023_Oct',
             month='october', year=2023),
        # 2024: August, November
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2024_Aug',
             month='august', year=2024),
        dict(col='DissCH4(µmol/liter)',
             units=r'CH$_4$ ($\mu$mol L$^{-1}$)',
             valmin=0, valmax=3.0,
             suffix='_WaterMethane_2024_Nov',
             month='november', year=2024),
        # ── Flux per-year (All months) ─────────────────────────────────────
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2023_All',
             month='all', year=2023),
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2024_All',
             month='all', year=2024),
        # ── Flux per-year per-month ────────────────────────────────────────
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2023_Jul',
             month='july', year=2023),
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2023_Aug',
             month='august', year=2023),
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2023_Oct',
             month='october', year=2023),
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2024_Aug',
             month='august', year=2024),
        dict(col='Flux (µmol/m2/day)',
             units=r'Flux ($\mu$mol m$^{-2}$ day$^{-1}$)',
             valmin=0, valmax=1000,
             suffix='_Flux_2024_Nov',
             month='november', year=2024),
    ]

    if run_mode == 'simple':
        variants = [v for v in variants if v['month'] == 'all' and v.get('year', 'all') == 'all']
        for v in variants:
            v['prefix'] = 'Simple_'

    for v in variants:
        mth = v['month']
        col = v['col']
        print(f'\n{"="*60}')
        print(f'Variant: {v.get("prefix", "")}{v["suffix"]}  |  col={col}  |  month={mth} | mode={run_mode}')
        print('='*60)
        run_variant(v, df_raw, da_bathy, dum, dum_da,
                    lon_g, lat_g, bathy_shape, land_mask, img_wms, coastlines)

    print('\n✅ All variants complete.')
    print(f'   Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
