# %%
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
import dunestyle.matplotlib as dunestyle

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

# %%
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

# stub functions
def mag(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def magdf(df):
    return mag(df.x, df.y, df.z)
PROTON_MASS = 0.938272

# Setup splines to map from Length or Charge to Energy
def dEdx2dQdx(dEdx):
    alpha = 0.930
    rho = 1.38434
    Efield = 0.5
    beta = 0.212 / (rho * Efield)
    Wion = 1e3 / 4.237e7
    return np.log(alpha + dEdx*beta) / (Wion*beta)

def dQdx2dEdx(dQdx):
    alpha = 0.930
    rho = 1.38434
    Efield = 0.5
    beta = 0.212 / (rho * Efield)
    Wion = 1e3 / 4.237e7
    return (np.exp(beta * Wion * dQdx) - alpha) / beta

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

# %%
range_spline = CubicSpline(RR, KE)
Q_spline = CubicSpline(integrated_dQ, KE)

# %%

def process_file(root_file):
    print(f"Processing file: {root_file}")
    try:
        events = uproot.open(f"{root_file}:recTree")
        print(f"Successfully opened {root_file}")
    except Exception as e:
        print(f"Error opening {root_file}: {e}")
        return

    print(f"Starting file: {root_file}")

    try:
        stubdf = loadbranches(events, stub_branches)
    except Exception as e:
        print(f"Error opening nudf {root_file}: {e}")
        return

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

    def dEdx2dQdx(dEdx, Efield=0.5):
        alpha = 0.930
        rho = 1.38434
        beta = 0.212 / (rho * Efield)
        Wion = 1e3 / 4.237e7
        return np.log(alpha + dEdx*beta) / (Wion*beta)

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

    flat_stubdf = stubdf.reset_index()
    output_file = f"./output/stubdf/{os.path.basename(root_file)}_stubdf.h5"
    print(f"Saving to {output_file}")
    flat_stubdf.to_hdf(output_file, key='stubdf', mode='w')
    print(f"Saved output to {output_file}")

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

