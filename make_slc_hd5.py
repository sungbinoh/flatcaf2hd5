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
    nudf['is_fv'] = is_fv
    nudf['is_signal'] = is_signal
    
    is_cccoh = CCCOH(nudf)
    nudf['CCCOH'] = is_cccoh

    print(f"Done mkaing flat df file: {root_file}")
    
    # %%
    flat_nudf = nudf.reset_index()
    output_file = f"./output/{os.path.basename(root_file)}_nudf.h5"
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

    # merge all
    masterdf = pd.merge(matchdf, pfpdf, left_index=True, right_index=True, how="inner")

    flat_masterdf = masterdf.reset_index()
    output_file = f"./output/{os.path.basename(root_file)}_masterdf.h5"
    print(f"Saving to {output_file}")
    flat_masterdf.to_hdf(output_file, key='masterdf', mode='w')
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
