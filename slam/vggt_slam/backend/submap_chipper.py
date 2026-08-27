#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, math
from pathlib import Path
import numpy as np, cv2

from vggt_slam.covis.dem_u8 import dem_to_uint8_gray

def load_global_contract(out_root: str):
    idx = json.load(open(os.path.join(out_root, "index.json"), "r"))
    U = np.asarray(idx["plane_U"], np.float64)
    V = np.asarray(idx["plane_V"], np.float64)
    N = np.asarray(idx["plane_N"], np.float64)
    d0 = float(idx.get("plane_d", 0.0))
    R2 = np.asarray(idx.get("pca_R2", [[1,0],[0,1]]), np.float64)
    mu = np.asarray(idx.get("pca_mu_xy", [0,0]), np.float64)
    grid = {
        "nx": idx["nx"], "ny": idx["ny"], "tile_px": idx["grid_size_px"],
        "mpp": idx["target_mpp"], "bbox": idx["bbox_global"],
        "clip_lo": idx.get("clip_lo", 1.0), "clip_hi": idx.get("clip_hi", 99.0),
        "kernel_px": idx.get("kernel_px", 1.2),
        "reducer": idx.get("reducer", "softmax"),
        "softmax_tau": idx.get("softmax_tau", 0.02),
    }
    return (U,V,N,d0,R2,mu,grid)

def world_to_plane_xy(P, U,V,N,d0,R2,mu):
    xy = np.stack([P @ U, P @ V], 1)
    xy = (xy - mu) @ R2.T
    z  = (P @ N) + d0
    return xy.astype(np.float32), z.astype(np.float32)

def rasterize_chip(xy, z, tb, mpp, reducer="softmax", tau=0.02, kernel_px=1.2):
    x0,y0,x1,y1 = tb
    W = max(1, int(math.ceil((x1-x0)/mpp)))
    H = max(1, int(math.ceil((y1-y0)/mpp)))
    dem = np.zeros((H,W), np.float32)
    occ = np.zeros((H,W), bool)

    px = ((xy[:,0]-x0)/mpp).astype(np.int32)
    py = ((xy[:,1]-y0)/mpp).astype(np.int32)
    inb = (px>=0)&(px<W)&(py>=0)&(py<H)
    px,py,zv = px[inb],py[inb],z[inb]
    if zv.size == 0: return dem, occ

    if reducer=="max":
        np.maximum.at(dem,(py,px),zv); occ[py,px]=True
    elif reducer=="mean":
        cnt = np.zeros((H,W), np.float32)
        np.add.at(dem,(py,px),zv); np.add.at(cnt,(py,px),1.0)
        occ = cnt>0; nz = occ; dem[nz] = dem[nz]/(cnt[nz]+1e-9)
    else:  # softmax
        zmax = np.full((H,W), -np.inf, np.float32)
        np.maximum.at(zmax,(py,px),zv)
        w = np.exp((zv - zmax[py,px]) / max(1e-6, tau)).astype(np.float32)
        sumw = np.zeros((H,W), np.float32); sumzw = np.zeros((H,W), np.float32)
        np.add.at(sumw,(py,px),w); np.add.at(sumzw,(py,px),zv*w)
        occ = sumw>0; nz = occ; dem[nz] = sumzw[nz]/(sumw[nz]+1e-12)

    # normalized Gaussian hole fill
    k = int(max(1, round(kernel_px))) * 2 + 1
    valid = occ.astype(np.float32)
    for _ in range(3):
        num = cv2.GaussianBlur(dem*valid,(k,k),kernel_px)
        den = cv2.GaussianBlur(valid,(k,k),kernel_px) + 1e-9
        dem[~occ] = (num/den)[~occ]
        valid = cv2.dilate(valid, np.ones((1,1), np.uint8), 1)
    return dem, occ

def chip_submap(out_root: str, sm_dir: str):
    U,V,N,d0,R2,mu,grid = load_global_contract(out_root)
    P_path = os.path.join(sm_dir, "points_world.npy")
    if not os.path.exists(P_path):
        print(f"[chip] no points: {P_path}")
        return 0
    P = np.load(P_path).astype(np.float32)
    if P.size == 0: return 0
    xy, z = world_to_plane_xy(P, U,V,N,d0,R2,mu)

    tiles_dir  = os.path.join(out_root, "tiles")
    chips_dir  = os.path.join(sm_dir, "chips")
    os.makedirs(chips_dir, exist_ok=True)

    metas = [os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir) if f.endswith(".meta.json")]
    metas.sort()
    c = 0
    for mp in metas:
        m = json.load(open(mp, "r"))
        tb = m["bbox"]; mpp = m.get("mpp", grid["mpp"])
        dem, occ = rasterize_chip(xy, z, tb, mpp,
                                  reducer=grid["reducer"],
                                  tau=grid["softmax_tau"],
                                  kernel_px=grid["kernel_px"])
        if np.any(occ):
            tid = int(m["id"])
            np.save(os.path.join(chips_dir, f"chip_{tid:05d}.npy"), dem.astype(np.float32))
            u8 = dem_to_uint8_gray(dem, clip_lo=grid["clip_lo"], clip_hi=grid["clip_hi"])
            cv2.imwrite(os.path.join(chips_dir, f"chip_{tid:05d}_u8.png"), u8)
            c += 1
    print(f"[chip] {c} chips written for {sm_dir}")
    return c

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--sm_dir", type=str, required=True)
    args = ap.parse_args()
    chip_submap(args.out_root, args.sm_dir)
