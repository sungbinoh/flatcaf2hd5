import os

import numpy as np
import math
import uproot as uproot
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib import gridspec
#import dunestyle.matplotlib as dunestyle

import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import median_abs_deviation
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import scipy.linalg as la
import scipy.optimize as opt
from scipy.optimize import Bounds, LinearConstraint
from scipy.stats import chisquare

from branches import *
from pandas_helpers import *

from multiprocessing import Pool

PDG = {
    "muon": [13, "muon", 0.105,],
    "proton": [2212, "proton", 0.938272,],
    "neutron": [2112, "neutron", 0.9395654,],
    "pizero": [111, "pizero", 0.1349768],
    "piplus": [211, "piplus", 0.13957039],
    "piminus": [-211, "piminus", 0.13957039],
    "argon": [1000180400, "argon", (18*0.938272 + 22*0.9395654)],
    "gamma": [22, "gamma", 0 ],
    "lambda": [3122, "lambda", 1.115683],
    "kaon_p": [321, "kaon_p",  0.493677],
    "sigma_p": [3222, "sigma_p", 1.18936],
    "kaon_0": [311, "kaon_0", 0.497648],
    "sigma_0": [3212, "sigma_0", 1.19246],
    "lambda_p_c": [4122, "lambda_p_c", 2.28646],
    "sigma_pp_c": [4222, "sigma_pp_c", 2.45397],
    "electron": [11, "electron", 0.510998950],
    "sigma_p_c": [4212, "sigma_p_c", 2.4529],
}

THRESHOLD = {"muon": 0.1, "proton_stub": 0.2, "picharged": 0.1, "pizero":0}

# %%
def InFV(data): # cm
    xmin = -199.15 + 10
    ymin = -200. + 10
    zmin = 0.0 + 10
    xmax = 199.15 - 10
    ymax =  200. - 10
    zmax =  500. - 50
    return (data.x > xmin) & (data.x < xmax) & (data.y > ymin) & (data.y < ymax) & (data.z > zmin) & (data.z < zmax)

def InBeam(t):
    return (t > 0.) & (t < 1.800)

def Avg(df, pid, drop_0=True):  # average score of 3 planes, exclude value if 0
    if drop_0:
        df = df.replace(0, np.nan)
    average = df[[("chi2pid", "I0", "chi2_"+pid), ("chi2pid", "I1", "chi2_"+pid), ("chi2pid", "I2", "chi2_"+pid)]].mean(skipna=drop_0, axis=1)
    return average

def dist_pfptrk_vertex(df):
    this_vertex_x = df[('slc', 'vertex', 'x')]
    this_vertex_y = df[('slc', 'vertex', 'y')]
    this_vertex_z = df[('slc', 'vertex', 'z')]

    this_pfp_start_x = df[('trk', 'start', 'x')]
    this_pfp_start_y = df[('trk', 'start', 'y')]
    this_pfp_start_z = df[('trk', 'start', 'z')]

    this_dist = np.sqrt(
        (this_vertex_x - this_pfp_start_x) ** 2 +
        (this_vertex_y - this_pfp_start_y) ** 2 +
        (this_vertex_z - this_pfp_start_z) ** 2
    )

    return this_dist

def Signal(df): # definition
    is_fv = InFV(df.nu.position)
    is_numu = (df.nu.pdg == 14) | (df.nu.pdg == -14)
    is_cc = (df.nu.iscc == 1)
    is_coh = (df.nu.genie_mode == 3)
    is_1pi0p = (df.mult_muon_def == 1) & (df.mult_picharged_def == 1) & (df.mult_proton_stub_def == 0)  & (df.mult_pizero == 0)
    return is_fv & is_numu & is_cc & is_1pi0p & is_coh

def CCCOH(df):
    is_cc = df.nu.iscc
    genie_mode = df.nu.genie_mode
    return is_cc & (genie_mode == 3)

def reco_t(n_trk_mupid, dir_x, dir_y, dir_z, range_P_muon, range_P_pion, mu_pid_pass):
    #print("reco_t")
    if n_trk_mupid != 2:
        return -999.  
    dir_x = dir_x[mu_pid_pass]
    dir_y = dir_y[mu_pid_pass]
    dir_z = dir_z[mu_pid_pass]
    range_P_muon = range_P_muon[mu_pid_pass]
    range_P_pion = range_P_pion[mu_pid_pass]
    if(range_P_muon.size != 2):
        print("error, dir_x.len != 2")
        return -888.
    
    # -- assume first particle is muon and the other is pion
    mass_0 = PDG["muon"][2]
    mass_1 = PDG["piplus"][2]
    p_0 = range_P_muon.iloc[0]
    p_1 = range_P_pion.iloc[1]
    # -- if second track is longer, swap the mass assumption
    if(range_P_muon.iloc[0] > range_P_muon.iloc[1]):
        mass_0 = PDG["piplus"][2]
        mass_1 = PDG["muon"][2]
        p_0 = range_P_pion.iloc[0]
        p_1 = range_P_muon.iloc[1]
    E_0 = np.sqrt(mass_0**2 + p_0**2)
    E_1 = np.sqrt(mass_1**2 + p_1**2)

    # -- each term
    px_sq = np.power(p_0 * dir_x.iloc[0] + p_1 * dir_x.iloc[1], 2.)
    py_sq = np.power(p_0 * dir_y.iloc[0] + p_1 * dir_y.iloc[1], 2.)
    pz_sq = np.power(E_0 + E_1 - p_0 * dir_z.iloc[0] - p_1 * dir_z.iloc[1], 2.)
    abs_t = px_sq + py_sq + pz_sq
    
    #print(abs_t)
    return abs_t

def measure_reco_t(group):
    n_trk_mupid = group[('n_trk_mupid', '', '')].iloc[0]
    dir_x = group[('trk', 'dir', 'x')]
    dir_y = group[('trk', 'dir', 'y')]
    dir_z = group[('trk', 'dir', 'z')]
    range_P_muon = group[('trk', 'rangeP', 'p_muon')]
    range_P_pion = group[('trk', 'rangeP', 'p_pion')]
    mu_pid_pass = group[('trk', 'mu_pid_pass', '')]

    # Call reco_t function
    return reco_t(n_trk_mupid, dir_x, dir_y, dir_z, range_P_muon, range_P_pion, mu_pid_pass)

def opening_angle(n_trk_mupid, dir_x, dir_y, dir_z, mu_pid_pass):
    #print("opening_angle")
    if n_trk_mupid != 2:
        return -999.
    dir_x = dir_x[mu_pid_pass]
    dir_y = dir_y[mu_pid_pass]
    dir_z = dir_z[mu_pid_pass]
    if(dir_x.size != 2):
        print("error, dir_x.len != 2")
        return -888.
    
    this_cos_theta = dir_x.iloc[0] * dir_x.iloc[1] + dir_y.iloc[0] * dir_y.iloc[1] + dir_z.iloc[0] * dir_z.iloc[1]
    return this_cos_theta
    
def measure_opening_angle(group):
    n_trk_mupid = group[('n_trk_mupid', '', '')].iloc[0]
    dir_x = group[('trk', 'dir', 'x')]
    dir_y = group[('trk', 'dir', 'y')]
    dir_z = group[('trk', 'dir', 'z')]
    mu_pid_pass = group[('trk', 'mu_pid_pass', '')]

    # Call reco_t function
    return opening_angle(n_trk_mupid, dir_x, dir_y, dir_z, mu_pid_pass)

def beam_totp_angle(n_trk_mupid, dir_x, dir_y, dir_z, range_P_muon, range_P_pion, mu_pid_pass):
    if n_trk_mupid != 2:
        return -999.  
    dir_x = dir_x[mu_pid_pass]
    dir_y = dir_y[mu_pid_pass]
    dir_z = dir_z[mu_pid_pass]
    range_P_muon = range_P_muon[mu_pid_pass]
    range_P_pion = range_P_pion[mu_pid_pass]
    if(range_P_muon.size != 2):
        print("error, dir_x.len != 2")
        return -888.
    
    # -- assume first particle is muon and the other is pion
    p_0 = range_P_muon.iloc[0]
    p_1 = range_P_pion.iloc[1]
    # -- if second track is longer, swap the mass assumption
    if(range_P_muon.iloc[0] > range_P_muon.iloc[1]):
        p_0 = range_P_pion.iloc[0]
        p_1 = range_P_muon.iloc[1]

    totpx = p_0 * dir_x.iloc[0] + p_1 * dir_x.iloc[1]
    totpy = p_0 * dir_y.iloc[0] + p_1 * dir_y.iloc[1]
    totpz = p_0 * dir_z.iloc[0] + p_1 * dir_z.iloc[1]

    totp_cos = totpz / np.power(np.power(totpx, 2.) + np.power(totpy, 2.) + np.power(totpz, 2.) , 0.5)
    return totp_cos
    
def measure_beam_totp_angle(group):
    n_trk_mupid = group[('n_trk_mupid', '', '')].iloc[0]
    dir_x = group[('trk', 'dir', 'x')]
    dir_y = group[('trk', 'dir', 'y')]
    dir_z = group[('trk', 'dir', 'z')]
    range_P_muon = group[('trk', 'rangeP', 'p_muon')]
    range_P_pion = group[('trk', 'rangeP', 'p_pion')]
    mu_pid_pass = group[('trk', 'mu_pid_pass', '')]

    # Call reco_t function
    return beam_totp_angle(n_trk_mupid, dir_x, dir_y, dir_z, range_P_muon, range_P_pion, mu_pid_pass)

def make_stub_df(events):
    PROTON_MASS = 0.938272
    def mag(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)
    
    def magdf(df):
        return mag(df.x, df.y, df.z)
    
    def dEdx2dQdx(dEdx):
        alpha = 0.930
        rho = 1.38434
        Efield = 0.5
        beta = 0.212 / (rho * Efield)
        Wion = 1e3 / 4.237e7
        return np.log(alpha + dEdx*beta) / (Wion*beta)
    
    def dEdx2dQdx(dEdx, Efield=0.5):
        alpha = 0.930
        rho = 1.38434
        beta = 0.212 / (rho * Efield)
        Wion = 1e3 / 4.237e7
        return np.log(alpha + dEdx*beta) / (Wion*beta)
    
    # Load the pstar data
    LAr_density = 1.38434
    data = np.genfromtxt('PSTAR.txt',dtype=None).T
    KE = data[0, :]
    dEdx = data[1,:] * LAr_density
    RR = data[2,:] / LAr_density
    dQdx = dEdx2dQdx(dEdx)
    
    RR0 = np.hstack([[0.], RR])
    integrated_dQ = np.cumsum(dQdx * (RR-RR0[:-1]))

    RR_dEdx_spline = CubicSpline(RR, dEdx)
    RR_many = np.logspace(np.log10(np.min(RR)), np.log10(np.max(RR)), 1000)
    dEdx_many = RR_dEdx_spline(RR_many)
    dQdx_many = dEdx2dQdx(dEdx_many)
    RR_many_0 = np.hstack([[0.], RR_many])
    integrated_dQ_many = np.cumsum(dQdx_many * (RR_many-RR_many_0[:-1]))
    KE_many = np.cumsum(dEdx_many * (RR_many-RR_many_0[:-1]))

    RRall = RR
    dQdxall = dQdx
    deltaQdeltaX = integrated_dQ / RRall

    where = (KE < 200) & (KE > 10)
    KE = KE[where]
    integrated_dQ = integrated_dQ[where]
    RR = RR[where]

    dQdx = dQdx[where]

    where_many = (KE_many < 200) & (KE_many > 10)
    KE_many = KE_many[where_many]
    integrated_dQ_many = integrated_dQ_many[where_many]
    RR_many = RR_many[where_many]

    max_delta_Q = np.max(integrated_dQ)
    min_delta_Q = np.min(integrated_dQ)
    
    def deltaQ2deltaE_with(deltaQ, fitfunc, popt):
        ret = fitfunc(deltaQ, *popt)
        ret[deltaQ < min_delta_Q] = -10000.
        ret[deltaQ > max_delta_Q] =  10000.
        return ret

    def deltaQ2deltaE(deltaQ):
        return deltaQ2deltaE_with(deltaQ, Q_spline, [])
    
    range_spline = CubicSpline(RR, KE)
    Q_spline = CubicSpline(integrated_dQ, KE)
    
    stubdf = loadbranches(events, stub_branches)
    stubdf = stubdf.rec.slc.reco.stub

    # stub dfs
    stubdf = loadbranches(events, stub_branches)
    stubdf = stubdf.rec.slc.reco.stub

    stubpdf = loadbranches(events, stub_plane_branches)
    stubpdf = stubpdf.rec.slc.reco.stub.planes

    stubdf["nplane"] = stubpdf.groupby(level=[0,1,2]).size()
    stubdf["plane"] = stubpdf.p.groupby(level=[0,1,2]).first()

    stubhitdf = loadbranches(events, stub_hit_branches)
    stubhitdf = stubhitdf.rec.slc.reco.stub.planes.hits

    stubhitdf = stubhitdf.join(stubpdf)
    stubhitdf = stubhitdf.join(stubdf.efield_vtx)
    stubhitdf = stubhitdf.join(stubdf.efield_end)

    MIP_dqdx = dEdx2dQdx(1.7) # dEdx2dQdx(1.8, (stubhitdf.efield_vtx + stubhitdf.efield_end) / 2.) * (0.01420 / 1.59e-2)

    stub_end_charge = stubhitdf.charge[stubhitdf.wire == stubhitdf.hit_w].groupby(level=[0,1,2,3]).first().groupby(level=[0,1,2]).first()
    stub_end_charge.name = ("endp_charge", "", "")

    stub_pitch = stubpdf.pitch.groupby(level=[0,1,2]).first()
    stub_pitch.name = ("pitch", "", "")

    stubdir_is_pos = (stubhitdf.hit_w - stubhitdf.vtx_w) > 0.
    when_sum = ((stubhitdf.wire > stubhitdf.vtx_w) == stubdir_is_pos) & (((stubhitdf.wire < stubhitdf.hit_w) == stubdir_is_pos) | (stubhitdf.wire == stubhitdf.hit_w)) 
    stubcharge = (stubhitdf.charge[when_sum]).groupby(level=[0,1,2,3]).sum().groupby(level=[0,1,2]).first()
    stubcharge.name = ("charge", "", "")

    stubinccharge = (stubhitdf.charge).groupby(level=[0,1,2,3]).sum().groupby(level=[0,1,2]).first()
    stubinccharge.name = ("inc_charge", "", "")

    hit_before_start = ((stubhitdf.wire < stubhitdf.vtx_w) == stubdir_is_pos)
    stub_inc_sub_charge = (stubhitdf.charge - MIP_dqdx*stubhitdf.ontrack*(~hit_before_start)*stubhitdf.trkpitch).groupby(level=[0,1,2,3]).sum().groupby(level=[0,1,2]).first()
    stub_inc_sub_charge.name = ("inc_sub_charge", "", "")

    stubdf["charge"] = stubcharge
    stubdf["inc_charge"] = stubinccharge
    stubdf["inc_sub_charge"] = stub_inc_sub_charge
    stubdf["endp_charge"] = stub_end_charge
    stubdf["pitch"] = stub_pitch
    stubdf["length"] = magdf(stubdf.vtx - stubdf.end)

    dqdx = stubdf.inc_sub_charge / stubdf.length
    length = stubdf.length
    hasstub = (length < 4.) & \
        (((length > 0.) & (dqdx > 5.5e5)) |\
            ((length > 0.5) & (dqdx > 3.5e5)) |\
            ((length > 1) & (dqdx > 3e5)) |\
            ((length > 2) & (dqdx > 2e5)))

    stubdf['pass_proton_stub'] = hasstub
    return stubdf

def process_file(root_file):
    print(f"Processing file: {root_file}")
    # Load events from ROOT file using uproot
    try:
        events = uproot.open(f"{root_file}:recTree")
        print(f"Successfully opened {root_file}")
    except Exception as e:
        print(f"Error opening {root_file}: {e}")
        return

    print(f"Starting file: {root_file}")
    
    # MC truth
    try:
        nudf = loadbranches(events, mc_branches)
    except Exception as e:
        print(f"Error opening nudf {root_file}: {e}")
        return

    nudf = nudf.rec.mc

    print(f"loaded nudf file: {root_file}")
    
    try:
        nuprimdf = loadbranches(events, mc_prim_branches)
    except Exception as e:
        print(f"Error opening nuprimdf {root_file}: {e}")
        return
    nuprimdf = nuprimdf.rec.mc.nu
    nuprimdf[("prim","totp","")] = np.sqrt((nuprimdf.prim.startp.x)**2+(nuprimdf.prim.startp.y)**2+(nuprimdf.prim.startp.z)**2) # |momentum| branch

    print(f"loaded nuprimdf file: {root_file}")
    
    # primary track multiplicity
    mult_muon = (nuprimdf.prim.pdg == 13).groupby(level=[0,1]).sum()
    mult_muon_def = ((nuprimdf.prim.pdg == 13) & (nuprimdf.prim.totp > THRESHOLD["muon"])).groupby(level=[0,1]).sum()
    mult_proton = (nuprimdf.prim.pdg ==2212).groupby(level=[0,1]).sum()
    mult_proton_stub_def = ((nuprimdf.prim.pdg == 2212) & (nuprimdf.prim.totp > THRESHOLD["proton_stub"])).groupby(level=[0,1]).sum()
    mult_picharged = (np.abs(nuprimdf.prim.pdg) == 211).groupby(level=[0,1]).sum()
    mult_picharged_def = ((np.abs(nuprimdf.prim.pdg) == 211) & (nuprimdf.prim.totp > THRESHOLD["picharged"])).groupby(level=[0,1]).sum()
    mult_pizero = (np.abs(nuprimdf.prim.pdg) == 111).groupby(level=[0,1]).sum()
    mult_pizero_def = ((np.abs(nuprimdf.prim.pdg) == 111) & (nuprimdf.prim.totp > THRESHOLD["pizero"])).groupby(level=[0,1]).sum()
    mult_electron = (nuprimdf.prim.pdg == 11).groupby(level=[0,1]).sum()
    mult_photon = (nuprimdf.prim.pdg == 22).groupby(level=[0,1]).sum()
    nudf['mult_muon'] = mult_muon
    nudf['mult_muon_def'] = mult_muon_def
    nudf['mult_proton'] = mult_proton
    nudf['mult_proton_stub_def'] = mult_proton_stub_def
    nudf['mult_picharged'] = mult_picharged
    nudf['mult_picharged_def'] = mult_picharged_def
    nudf['mult_pizero'] = mult_pizero
    nudf['mult_pizero_def'] = mult_pizero_def
    nudf['mult_electron'] = mult_electron
    nudf['mult_photon'] = mult_photon

    print(f"added par mult cols file: {root_file}")
    
    try :
        nuint_categ = pd.Series(8, index=nudf.index)
        print(f"done init nuint_categ file: {root_file}")
    except Exception as e:
        print(f"Error init nuint_categ {root_file}: {e}")
        return

    # %%
    is_fv = InFV(nudf.nu.position)
    is_signal = Signal(nudf)
    is_cc = nudf.nu.iscc
    genie_mode = nudf.nu.genie_mode
    w = nudf.nu.w
    
    nuint_categ[~is_fv] = -1  # Out of FV
    nuint_categ[is_fv & ~is_cc] = 0  # NC
    nuint_categ[is_fv & is_cc & is_signal] = 1  # Signal
    nuint_categ[is_fv & is_cc & ~is_signal & (genie_mode == 3)] = 2  # Non-signal CCCOH
    nuint_categ[is_fv & is_cc & (genie_mode == 0)] = 3  # CCQE
    nuint_categ[is_fv & is_cc & (genie_mode == 10)] = 4  # 2p2h
    nuint_categ[is_fv & is_cc & (genie_mode != 0) & (genie_mode != 3) & (genie_mode != 10) & ((w < 1.4) | (genie_mode == 1))] = 5  # RES
    nuint_categ[is_fv & is_cc & (genie_mode != 0) & (genie_mode != 3) & (genie_mode != 10) & ((w > 2.0) | (genie_mode == 2))] = 6  # DIS
    nuint_categ[is_fv & is_cc & ((1.4 < w) & (w < 2.0) & (genie_mode != 1) & (genie_mode != 2) & (genie_mode != 0) & (genie_mode != 3) & (genie_mode != 10))] = 7  # INEL

    print(f"done making nuint categ serieses par mult cols file: {root_file}")
    
    nudf['nuint_categ'] = nuint_categ
    nudf['is_true_fv'] = is_fv
    nudf['is_true_signal'] = is_signal
    
    is_cccoh = CCCOH(nudf)
    nudf['CCCOH'] = is_cccoh

    #print(f"Done mkaing flat df file: {root_file}")
    
    # %%
    flat_nudf = nudf.reset_index()
    output_file = f"./output/nudf/{os.path.basename(root_file)}_nudf.h5"
    print(f"Saving to {output_file}")
    flat_nudf.to_hdf(output_file, key='nudf', mode='w')
    print(f"Saved output to {output_file}")

    # truth match
    slcdf = loadbranches(events, slc_branches)
    slcdf = slcdf.rec

    slcdf.loc[np.invert(slcdf[("slc","tmatch","eff")] > 0.5) & (slcdf[("slc","tmatch","idx")] >= 0), ("slc","tmatch","idx")] = np.nan
    slcdf["tmatch_index"] = slcdf[("slc", "tmatch", "idx")]
    matchdf = pd.merge(slcdf.reset_index(), 
                       nudf.reset_index(),
                       left_on=[("entry", "",""), ("slc","tmatch", "idx")], # entry index -> neutrino index
                       right_on=[("entry", "",""), ("rec.mc.nu..index", "","")], 
                       how="left", # Keep every slc
                       ) 
    matchdf = matchdf.set_index(["entry", "rec.slc..index"], verify_integrity=True)

    # reco pfps
    pfptrkdf = loadbranches(events, pfp_trk_branches)
    pfptrkdf = pfptrkdf.rec.slc.reco.pfp
    pfptrkchi2df = loadbranches(events, pfp_trk_chi2_branches)
    pfptrkchi2df = pfptrkchi2df.rec.slc.reco.pfp.trk
    pfptrkdf = pfptrkdf.join(pfptrkchi2df)

    pfptruthdf = loadbranches(events, pfp_trk_mc_branches)
    pfptruthdf = pfptruthdf.rec.slc.reco.pfp.trk.truth
    pfpdf = pd.merge(pfptrkdf, pfptruthdf, left_index=True, right_index=True, how="inner")

    pandoradf = loadbranches(events, pandora_branches)
    pandoradf = pandoradf.rec.slc
    cnniddf = loadbranches(events, cnn_branches)
    cnniddf = cnniddf.rec.slc.reco
    scoresdf = pd.merge(pandoradf, cnniddf, left_index=True, right_index=True, how="inner")
    pfpdf = pd.merge(pfpdf, scoresdf, left_index=True, right_index=True, how="inner")

    # event selections
    ## -- FV
    is_reco_fv = InFV(matchdf.slc.vertex)
    matchdf['is_reco_fv'] = is_reco_fv
    ## -- number of tracks with length > 4cm
    cut_trk_len = pfpdf.trk.len > 4.
    pfpdf[('trk', 'len', 'pass')] = cut_trk_len
    n_trk_df = cut_trk_len.reset_index(name='len')
    all_combinations = (
        n_trk_df[['entry', 'rec.slc..index']].drop_duplicates().set_index(['entry', 'rec.slc..index'])
    )
    n_trk_df = (
        n_trk_df[n_trk_df['len'] == True]
        .groupby(['entry', 'rec.slc..index'])
        .size()
        .reindex(all_combinations.index, fill_value=0)
    )
    matchdf['n_trk_4cm'] = n_trk_df
    
    # -- count pfp tracks passing track-len and vtx dist cut
    masterdf = pd.merge(matchdf, pfpdf, left_index=True, right_index=True, how="inner")
    this_df_series = dist_pfptrk_vertex(masterdf)
    masterdf['dist_pfptrk_vertex'] = this_df_series
    cut_vtx_dist = masterdf.dist_pfptrk_vertex < 6.
    cut_vtx_dist = cut_trk_len & cut_vtx_dist
    pfpdf[('trk', 'vtxdist', 'pass')] = cut_vtx_dist
    n_pass_vtxdist = cut_vtx_dist.reset_index(name='vtxdist')
    n_pass_vtxdist = (
        n_pass_vtxdist[n_pass_vtxdist['vtxdist'] == True]
        .groupby(['entry', 'rec.slc..index'])
        .size()
        .reindex(all_combinations.index, fill_value=0)
    )
    matchdf['n_trk_vtxdist'] = n_pass_vtxdist

    # -- count pfp tracks passing chi2 PID cut too
    cut_pidscore = (Avg(pfpdf, "muon", drop_0=True) < 25) & (Avg(pfpdf, "proton", drop_0=True) > 100)
    cut_pidscore = cut_pidscore & cut_trk_len & cut_vtx_dist
    pfpdf[('trk', 'mu_pid_pass', '')] = cut_pidscore
    n_pass_mupid_df = cut_pidscore.reset_index(name='pidscore')
    n_pass_mupid_df = (
        n_pass_mupid_df[n_pass_mupid_df['pidscore'] == True]
        .groupby(['entry', 'rec.slc..index'])
        .size()
        .reindex(all_combinations.index, fill_value=0)
    )
    matchdf['n_trk_mupid'] = n_pass_mupid_df

    masterdf = pd.merge(matchdf, pfpdf, left_index=True, right_index=True, how="inner")
    reco_t_series = masterdf.groupby(['entry', 'rec.slc..index']).apply(measure_reco_t)
    reco_t_df = reco_t_series.to_frame(name='reco_t_value')
    reco_t_df.index.set_names(['entry', 'rec.slc..index'], inplace=True)
    matchdf['reco_t'] = reco_t_df

    opening_angle_series = masterdf.groupby(['entry', 'rec.slc..index']).apply(measure_opening_angle)
    opening_angle_df = opening_angle_series.to_frame(name='reco_opening_angle')
    opening_angle_df.index.set_names(['entry', 'rec.slc..index'], inplace=True)
    matchdf['opening_angle'] = opening_angle_df

    beam_totp_angle_series = masterdf.groupby(['entry', 'rec.slc..index']).apply(measure_beam_totp_angle)
    beam_totp_angle_df = beam_totp_angle_series.to_frame(name='reco_beam_totp_angle')
    beam_totp_angle_df.index.set_names(['entry', 'rec.slc..index'], inplace=True)
    matchdf['beam_totp_angle'] = beam_totp_angle_df
    
    stubdf = make_stub_df(events)
    cut_stub_proton = stubdf.pass_proton_stub
    n_stub_proton_df = cut_stub_proton.reset_index(name='pass_proton_stub')
    all_combinations = (
        n_stub_proton_df[['entry', 'rec.slc..index']].drop_duplicates().set_index(['entry', 'rec.slc..index'])
    )
    n_stub_proton_df = (
        n_stub_proton_df[n_stub_proton_df['pass_proton_stub'] == True]
        .groupby(['entry', 'rec.slc..index'])
        .size()
        .reindex(all_combinations.index, fill_value=0)
    )
    n_stub_series = pd.Series(-1, index=matchdf.index)
    n_stub_series.update(n_stub_proton_df)
    matchdf["n_stub_proton"] = n_stub_series

    ## save matchdf
    flat_matchdf = matchdf.reset_index()
    output_file = f"./output/matchdf/{os.path.basename(root_file)}_matchdf.h5"
    print(f"Saving to {output_file}")
    flat_matchdf.to_hdf(output_file, key='matchdf', mode='w')
    print(f"Saved output to {output_file}")
    
# Main function to process multiple files using multithreading
def main(file_list_path):
    # Read the list of files from the provided text file
    with open(file_list_path, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)

    # Use ThreadPoolExecutor for multithreading
    with Pool() as pool:
        pool.map(process_file, file_list)

if __name__ == "__main__":
    # Text file containing the list of ROOT files
    file_list_path = "./root_files_list.txt"
    main(file_list_path)
