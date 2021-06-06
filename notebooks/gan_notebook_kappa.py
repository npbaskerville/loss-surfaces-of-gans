#!/usr/bin/env python
# coding: utf-8

# # Varying GAN hyperparameters
#
# We analyse results of experiments training real GANs (DCGAN) on CIFAR10 for various choices of the noise variance parameter $\kappa_z$ and $\kappa$.
#
# Experiments are conducted outside the notebook using the `gan.py` script.

# In[1]:


from collections import defaultdict
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


figsize = (5, 5)

def multiply_figsize(fs_minor, fs_major):
    return (fs_minor[0]*fs_major[0], fs_minor[1]*fs_major[1])


# In[3]:

#
# top_results_loc = "/work/jr19127/gan-loss-surfaces/rmt_results/vary_kappa/"
# vary_pq_kappa_results = {}
# for subdir in tqdm(os.listdir(top_results_loc)):
#     try:
#         results_loc = os.path.join(top_results_loc, subdir)
#         result_fns = sorted(os.listdir(results_loc), key=lambda s: int(s.split("_")[1]))
#         rmt_results = []
#         for fn in result_fns:
#             with open(os.path.join(results_loc, fn), "rb") as fin:
#                 rmt_results.append([x if x is not None else np.nan for x in pkl.load(fin)])
#         rmt_results = np.array(rmt_results)
#
#         kappas = rmt_results[:, 0]
#         _min_uDs = rmt_results[:, 1]
#         _min_uGs = rmt_results[:, 2]
#         min_sums = rmt_results[:, 3]
#         max_diffs = rmt_results[:, 4]
#
#         vary_pq_kappa_results[subdir] = [_min_uDs, _min_uGs]
#     except:
#         continue
#
#
# # In[4]:
#
#
# min_uDs, min_uGs = vary_pq_kappa_results["p5q5"]
#
#
# # In[5]:
#
#
# results_dir = "/work/jr19127/gan-loss-surfaces/vary_kappa_dcgan_cifar10"
# results_dirs  = [os.path.join(results_dir, "results_{}".format(ind))
#                  for ind in range(len(os.listdir(results_dir)))][:-1]
# pkl_files = [os.path.join(rdir, x) for rdir in results_dirs for x in os.listdir(rdir) if x[-3:]==".pk"]
#
# results = defaultdict(list)
# for fn in tqdm(pkl_files):
#     with open(os.path.join(results_dir, fn), "rb") as fin:
#         results[float(fn.split("/")[-1][:-3])].append(np.array(pkl.load(fin)))
#
#
# # In[ ]:
#
#
# def summary(arr):
#     return min(arr)
#
# discrims =np.array([np.mean([summary(r[0]) for r in results[s]]) for s in kappas])
# discrims_std = np.array([np.std([summary(r[0]) for r in results[s]]) for s in kappas])
#
# gens = np.array([np.mean([summary(r[1]) for r in results[s]]) for s in kappas])
# gens_std = np.array([np.std([summary(r[1]) for r in results[s]]) for s in kappas])
#
#
# # In[ ]:
#
#
# min_kappa = kappas[~np.isnan(min_uGs)].min()
#
#
# # In[ ]:
#
#
# fig = plt.figure(figsize=multiply_figsize(figsize, (2, 1)))
# plot_reals = gens
# plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
# plot_reals = plot_reals[kappas >= min_kappa]
# plot_rmts = min_uGs[kappas >= min_kappa]
#
# varname= "L_G"
# theory_name = "\\theta_G"
#
# plt.subplot(1, 2, 1);
# plt.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, );
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=20)
# yticks = plt.yticks(fontsize=20)
# plt.ylabel("min ${}$".format(varname), fontsize=20);
# plt.subplot(1, 2, 2);
# plt.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3);
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=20)
# yticks = plt.yticks(fontsize=20)
# plt.ylabel("${}$".format(theory_name), fontsize=20);
# plt.tight_layout();
#
# plt.savefig("../figures/real_gan_vs_theory_{}.pdf".format(varname))
#
#
# # In[ ]:
#
#
# fig, ax1= plt.subplots(figsize=multiply_figsize(figsize, (1, 1)))
# plot_reals = gens
# plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
# plot_reals = plot_reals[kappas >= min_kappa]
# plot_rmts = min_uGs[kappas >= min_kappa]
#
# varname= "L_G"
# theory_name = "\\theta_G"
#
# # plt.subplot(1, 2, 1);
# ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_G$");
# ax1.set_xlabel("$\kappa_z$", fontsize=20);
# ax1.set_ylabel("min ${}$".format(varname), fontsize=20);
#
# ax = ax1.twinx()
# # plt.subplot(1, 2, 2);
# ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_G$");
#
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=20)
# # yticks = ax.set_yticks(fontsize=20)
#
# ax.set_ylabel("${}$".format(theory_name), fontsize=20);
# plt.tight_layout();
#
# lns = ln1+ln2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
# plt.savefig("../figures/real_gan_vs_theory_pair_{}.pdf".format(varname))
#
#
# # Some experiments varying $p$ and $q$ in the RMT calculations.
#
# # In[ ]:
#
#
# top_results_loc = "/work/jr19127/gan-loss-surfaces/rmt_results/vary_kappa_variety_params"
# vary_pq_kappa_results = {}
# vary_pq_kappa_results["p5q5"] = [min_uDs, min_uGs]
# for subdir in tqdm(os.listdir(top_results_loc)):
#     if subdir == "p10q10":
#         continue
#     results_loc = os.path.join(top_results_loc, subdir)
#     result_fns = sorted(os.listdir(results_loc), key=lambda s: int(s.split("_")[1]))
#     rmt_results = []
#     for fn in result_fns:
#         with open(os.path.join(results_loc, fn), "rb") as fin:
#             rmt_results.append([x if x is not None else np.nan for x in pkl.load(fin)])
#     rmt_results = np.array(rmt_results)
#
#     kappas = rmt_results[:, 0]
#     _min_uDs = rmt_results[:, 1]
#     _min_uGs = rmt_results[:, 2]
#     min_sums = rmt_results[:, 3]
#     max_diffs = rmt_results[:, 4]
#
#     vary_pq_kappa_results[subdir] = [_min_uDs, _min_uGs]
#
#
# # In[ ]:
#
#
# def parse_name(name):
#     a, b = name.split("q")
#     q = int(b)
#     p = int(a[1:])
#     return f"p={p}, q={q}"
#
# fig = plt.figure(figsize=multiply_figsize(figsize, (1, 1)))
# plt.subplot(1, 1, 1);
# styles = ['-', '-.', ':',  ] * 4
# markers = ['o', 's', 'd', '*']*2
# for style, marker, (exp, results) in zip(styles, markers, vary_pq_kappa_results.items()):
# #     if exp == "p3q3":
# #         continue
#     _, min_uGs = results
#     plot_reals = gens
#     plot_rmts = min_uGs[kappas >= min_kappa]
#     varname= "L_G"
#     theory_name = "\\theta_G"
#
#     plt.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, label=parse_name(exp), linestyle=style);
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=10)
# yticks = plt.yticks(fontsize=10)
# plt.ylabel("${}$".format(theory_name), fontsize=20);
# plt.legend();
#
# plt.ylabel("min ${}$".format(varname), fontsize=20);
#
# plt.tight_layout();
#
# plt.savefig("../figures/gan_theory_{}_vary_pq.pdf".format(varname))
#
#
# # In[ ]:
#
#
# fig, axs= plt.subplots(1, len(vary_pq_kappa_results), figsize=multiply_figsize(figsize, (len(vary_pq_kappa_results), 1)))
#
# for ind, (exp, results) in enumerate(vary_pq_kappa_results.items()):
#     _, min_uGs = results
#     plot_reals = gens
#     plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
#     plot_reals = plot_reals[kappas >= min_kappa]
#     plot_rmts = min_uGs[kappas >= min_kappa]
#
#     varname= "L_G"
#     theory_name = "\\theta_G"
#     ax1 = axs[ind]
#     # plt.subplot(1, 2, 1);
#     ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_G$");
#     ax1.set_xlabel("$\kappa_z$", fontsize=20);
#     ax1.set_ylabel("min ${}$".format(varname), fontsize=20);
#
#     ax = ax1.twinx()
#     # plt.subplot(1, 2, 2);
#     ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_G$");
#
#     plt.xlabel("$\kappa_z$", fontsize=20);
#     plt.xscale('log')
#     xticks = plt.xticks(fontsize=20)
#     # yticks = ax.set_yticks(fontsize=20)
#
#     ax.set_ylabel("${}$".format(theory_name), fontsize=20);
#
#     lns = ln1+ln2
#     labs = [l.get_label() for l in lns]
#     ax.legend(lns, labs, loc=0)
#     ax.set_title(parse_name(exp))
# plt.tight_layout();
# plt.savefig("../figures/real_gan_vs_theory_pair_variety{}.pdf".format(varname))
#
#
# # In[ ]:
#
#
# fig = plt.figure(figsize=multiply_figsize(figsize, (2, 1)))
# plot_reals = discrims
# plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
# max_kappa = np.max(kappas[~np.isnan(plot_reals)])
# plot_reals = plot_reals[(kappas >= min_kappa) & (kappas <= max_kappa)]
# plot_rmts = min_uDs[(kappas >= min_kappa) & (kappas <= max_kappa)]
#
# varname= "L_D"
# theory_name = "\\theta_D"
#
# plt.subplot(1, 2, 1);
# plt.plot(kappas[(kappas >= min_kappa) & (kappas <= max_kappa)], plot_reals, 'x--', markersize=10, );
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=20)
# plt.xticks([1e-4, 1e-2, 1, 1e2])
# yticks = plt.yticks(fontsize=20)
# plt.ylabel("min ${}$".format(varname), fontsize=20);
# plt.subplot(1, 2, 2);
# plt.plot(kappas[(kappas >= min_kappa) & (kappas <= max_kappa)], plot_rmts, linewidth=3);
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=20)
# plt.xticks([1e-4, 1e-2, 1, 1e2])
# yticks = plt.yticks(fontsize=20)
# plt.ylabel("${}$".format(theory_name), fontsize=20);
# plt.tight_layout();
#
# plt.savefig("../figures/real_gan_vs_theory_{}.pdf".format(varname))
#
#
# # In[ ]:
#
#
# fig, ax1= plt.subplots(figsize=multiply_figsize(figsize, (1, 1)))
# plot_reals = discrims
# plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
# plot_reals = plot_reals[kappas >= min_kappa]
# plot_rmts = min_uDs[kappas >= min_kappa]
#
# varname= "L_D"
# theory_name = "\\theta_D"
#
# # plt.subplot(1, 2, 1);
# ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_D$");
# ax1.set_xlabel("$\kappa_z$", fontsize=20);
# ax1.set_ylabel("min ${}$".format(varname), fontsize=20);
#
# ax = ax1.twinx()
# # plt.subplot(1, 2, 2);
# ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_D$");
#
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=20)
# # yticks = ax.set_yticks(fontsize=20)
#
# ax.set_ylabel("${}$".format(theory_name), fontsize=20);
# plt.tight_layout();
#
# lns = ln1+ln2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
# plt.savefig("../figures/real_gan_vs_theory_pair_{}.pdf".format(varname))
#
#
# # Some experiments varying $p$ and $q$ in the RMT calculations.
#
# # In[ ]:
#
#
# def parse_name(name):
#     a, b = name.split("q")
#     q = int(b)
#     p = int(a[1:])
#     return f"p={p}, q={q}"
#
# fig = plt.figure(figsize=multiply_figsize(figsize, (1, 1)))
# plt.subplot(1, 1, 1);
# styles = ['-', '-.', ':',  ] * 4
# markers = ['o', 's', 'd', '*']*2
# for style, marker, (exp, results) in zip(styles, markers, vary_pq_kappa_results.items()):
# #     if exp == "p3q3":
# #         continue
#     min_uDs, _ = results
#     plot_reals = discrims
#     plot_rmts = min_uDs[kappas >= min_kappa]
#     varname= "L_D"
#     theory_name = "\\theta_D"
#
#     plt.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, label=parse_name(exp), linestyle=style);
# plt.xlabel("$\kappa_z$", fontsize=20);
# plt.xscale('log')
# xticks = plt.xticks(fontsize=10)
# yticks = plt.yticks(fontsize=10)
# plt.ylabel("${}$".format(theory_name), fontsize=20);
# plt.legend();
#
# plt.ylabel("min ${}$".format(varname), fontsize=20);
#
# plt.tight_layout();
#
# plt.savefig("../figures/gan_theory_{}_vary_pq.pdf".format(varname))
#
#
# # In[ ]:
#
#
# fig, axs= plt.subplots(1, len(vary_pq_kappa_results), figsize=multiply_figsize(figsize, (len(vary_pq_kappa_results), 1)))
#
# for ind, (exp, results) in enumerate(vary_pq_kappa_results.items()):
#     min_uDs, _ = results
#     plot_reals = discrims
#     plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
#     plot_reals = plot_reals[kappas >= min_kappa]
#     plot_rmts = min_uDs[kappas >= min_kappa]
#
#     varname= "L_D"
#     theory_name = "\\theta_D"
#     ax1 = axs[ind]
#     # plt.subplot(1, 2, 1);
#     ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_D$");
#     ax1.set_xlabel("$\kappa_z$", fontsize=20);
#     ax1.set_ylabel("min ${}$".format(varname), fontsize=20);
#
#     ax = ax1.twinx()
#     # plt.subplot(1, 2, 2);
#     ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_D$");
#
#     plt.xlabel("$\kappa_z$", fontsize=20);
#     plt.xscale('log')
#     xticks = plt.xticks(fontsize=20)
#     # yticks = ax.set_yticks(fontsize=20)
#
#     ax.set_ylabel("${}$".format(theory_name), fontsize=20);
#
#     lns = ln1+ln2
#     labs = [l.get_label() for l in lns]
#     ax.legend(lns, labs, loc=0)
#     ax.set_title(parse_name(exp))
# plt.tight_layout();
# plt.savefig("../figures/real_gan_vs_theory_pair_variety{}.pdf".format(varname))


# In[ ]:


# In[ ]:



top_results_loc = "/work/jr19127/gan-loss-surfaces/rmt_results/vary_kappa/"
vary_pq_kappa_results = {}
for subdir in tqdm(os.listdir(top_results_loc)):
    try:
        results_loc = os.path.join(top_results_loc, subdir)
        result_fns = sorted(os.listdir(results_loc), key=lambda s: int(s.split("_")[1]))
        rmt_results = []
        for fn in result_fns:
            with open(os.path.join(results_loc, fn), "rb") as fin:
                rmt_results.append([x if x is not None else np.nan for x in pkl.load(fin)])
        rmt_results = np.array(rmt_results)

        kappas = rmt_results[:, 0]
        _min_uDs = rmt_results[:, 1]
        _min_uGs = rmt_results[:, 2]
        min_sums = rmt_results[:, 3]
        max_diffs = rmt_results[:, 4]

        vary_pq_kappa_results[subdir] = [_min_uDs, _min_uGs]
    except:
        continue


# In[4]:


min_uDs, min_uGs = vary_pq_kappa_results["p5q5"]

# In[ ]:
results_dir = "/work/jr19127/gan-loss-surfaces/vary_kappa_dcgan_cifar10_back"
results_dirs  = [os.path.join(results_dir, "results_{}".format(ind))
                 for ind in range(len(os.listdir(results_dir)))][:-1]
pkl_files = [os.path.join(rdir, x) for rdir in results_dirs for x in os.listdir(rdir) if x[-3:]==".pk"]

results = defaultdict(list)
for fn in tqdm(pkl_files):
    with open(os.path.join(results_dir, fn), "rb") as fin:
        results[float(fn.split("/")[-1][:-3])].append(np.array(pkl.load(fin)))

# def summary(arr):
# #     return np.min(pd.Series(arr).rolling(100).mean().dropna().values)
# #     return arr[-int(len(arr)*0.05):].mean()
#     return min(arr)
# #     return arr[-1]
# #     return np.min(pd.Series(arr).rolling(500).mean().dropna().values)


# discrims =np.array([np.mean([summary(r[0]) for r in results[s]]) for s in kappas])
# discrims_std = np.array([np.std([summary(r[0]) for r in results[s]]) for s in kappas])

# gens = np.array([np.mean([summary(r[1]) for r in results[s]]) for s in kappas])
# gens_std = np.array([np.std([summary(r[1]) for r in results[s]]) for s in kappas])

def summary(arr):
    return min(arr)

discrims =np.array([np.mean([summary(r[0]) for r in results[s]]) for s in kappas])
discrims_std = np.array([np.std([summary(r[0]) for r in results[s]]) for s in kappas])

gens = np.array([np.mean([summary(r[1]) for r in results[s]]) for s in kappas])
gens_std = np.array([np.std([summary(r[1]) for r in results[s]]) for s in kappas])


# In[ ]:


min_kappa = kappas[~np.isnan(min_uDs)].min()


# In[ ]:


fig = plt.figure(figsize=multiply_figsize(figsize, (2, 1)))
plot_reals = gens
plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
plot_reals = plot_reals[kappas >= min_kappa]
plot_rmts = min_uGs[kappas >= min_kappa]

varname= "L_G"
theory_name = "\\theta_G"

plt.subplot(1, 2, 1);
plt.plot(kappas, plot_reals, 'x--', markersize=10, );
plt.xlabel("$\kappa$", fontsize=20);
xticks = plt.xticks(fontsize=20)
yticks = plt.yticks(fontsize=20)
plt.ylabel("min ${}$".format(varname), fontsize=20);
plt.subplot(1, 2, 2);
plt.plot(kappas, plot_rmts, linewidth=3);
plt.xlabel("$\kappa$", fontsize=20);
xticks = plt.xticks(fontsize=20)
yticks = plt.yticks(fontsize=20)
plt.ylabel("${}$".format(theory_name), fontsize=20);
plt.tight_layout();

plt.savefig("../figures/real_gan_vs_theory_kappa_{}.pdf".format(varname))


# In[ ]:


fig, ax1= plt.subplots(figsize=multiply_figsize(figsize, (1, 1)))
plot_reals = gens
plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
plot_reals = plot_reals[kappas >= min_kappa]
plot_rmts = min_uGs[kappas >= min_kappa]

varname= "L_G"
theory_name = "\\theta_G"

# plt.subplot(1, 2, 1);
ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_G$");
ax1.set_xlabel("$\kappa$", fontsize=20);
ax1.set_ylabel("min ${}$".format(varname), fontsize=20);

ax = ax1.twinx()
# plt.subplot(1, 2, 2);
ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_G$");

plt.xlabel("$\kappa$", fontsize=20);
plt.xscale('log')
xticks = plt.xticks(fontsize=20)
# yticks = ax.set_yticks(fontsize=20)

ax.set_ylabel("${}$".format(theory_name), fontsize=20);
plt.tight_layout();

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.savefig("../figures/real_gan_vs_theory_pair_kappa_{}.pdf".format(varname))


# In[ ]:





# Some experiments varying $p$ and $q$ in the RMT calculations.

# In[ ]:


def parse_name(name):
    a, b = name.split("q")
    q = int(b)
    p = int(a[1:])
    return f"p={p}, q={q}"

fig = plt.figure(figsize=multiply_figsize(figsize, (1, 1)))
plt.subplot(1, 1, 1);
styles = ['-', '-.', ':',  ] * 4
markers = ['o', 's', 'd', '*']*2
for style, marker, (exp, results) in zip(styles, markers, vary_pq_kappa_results.items()):
#     if exp == "p3q3":
#         continue
    _, min_uGs = results
    plot_reals = gens
    plot_rmts = min_uGs[kappas >= min_kappa]
    varname= "L_D"
    theory_name = "\\theta_D"

    plt.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, label=parse_name(exp), linestyle=style);
plt.xlabel("$\kappa$", fontsize=20);
plt.xscale('log')
xticks = plt.xticks(fontsize=10)
yticks = plt.yticks(fontsize=10)
plt.ylabel("${}$".format(theory_name), fontsize=20);
plt.legend();

plt.ylabel("min ${}$".format(varname), fontsize=20);

plt.tight_layout();

plt.savefig("../figures/gan_theory_kappa_{}_vary_pq.pdf".format(varname))


# In[ ]:


fig, axs= plt.subplots(1, len(vary_pq_kappa_results), figsize=multiply_figsize(figsize, (len(vary_pq_kappa_results), 1)))

for ind, (exp, results) in enumerate(vary_pq_kappa_results.items()):
    _, min_uGs = results
    plot_reals = discrims
    plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
    plot_reals = plot_reals[kappas >= min_kappa]
    plot_rmts = min_uGs[kappas >= min_kappa]

    varname= "L_G"
    theory_name = "\\theta_G"
    ax1 = axs[ind]
    # plt.subplot(1, 2, 1);
    ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_G$");
    ax1.set_xlabel("$\kappa$", fontsize=20);
    ax1.set_ylabel("min ${}$".format(varname), fontsize=20);

    ax = ax1.twinx()
    # plt.subplot(1, 2, 2);
    ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_G$");

    plt.xlabel("$\kappa$", fontsize=20);
    plt.xscale('log')
    xticks = plt.xticks(fontsize=20)
    # yticks = ax.set_yticks(fontsize=20)

    ax.set_ylabel("${}$".format(theory_name), fontsize=20);

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.set_title(parse_name(exp))
plt.tight_layout();
plt.savefig("../figures/real_gan_vs_theory_kappa_pair_variety{}.pdf".format(varname))


# In[ ]:


fig = plt.figure(figsize=multiply_figsize(figsize, (2, 1)))
plot_reals = discrims
plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
plot_reals = plot_reals[kappas >= min_kappa]
plot_rmts = min_uDs[kappas >= min_kappa]

varname= "L_D"
theory_name = "\\theta_D"

plt.subplot(1, 2, 1);
plt.plot(kappas, plot_reals, 'x--', markersize=10, );
plt.xlabel("$\kappa$", fontsize=20);
xticks = plt.xticks(fontsize=20)
yticks = plt.yticks(fontsize=20)
plt.ylabel("min ${}$".format(varname), fontsize=20);
plt.subplot(1, 2, 2);
plt.plot(kappas, plot_rmts, linewidth=3);
plt.xlabel("$\kappa$", fontsize=20);
xticks = plt.xticks(fontsize=20)
yticks = plt.yticks(fontsize=20)
plt.ylabel("${}$".format(theory_name), fontsize=20);
plt.tight_layout();

plt.savefig("../figures/real_gan_vs_theory_kappa_{}.pdf".format(varname))


# In[ ]:


fig, ax1= plt.subplots(figsize=multiply_figsize(figsize, (1, 1)))
plot_reals = discrims
plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
plot_reals = plot_reals[kappas >= min_kappa]
plot_rmts = min_uDs[kappas >= min_kappa]

varname= "L_D"
theory_name = "\\theta_D"

# plt.subplot(1, 2, 1);
ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_D$");
ax1.set_xlabel("$\kappa$", fontsize=20);
ax1.set_ylabel("min ${}$".format(varname), fontsize=20);

ax = ax1.twinx()
# plt.subplot(1, 2, 2);
ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_D$");

plt.xlabel("$\kappa$", fontsize=20);
plt.xscale('log')
xticks = plt.xticks(fontsize=20)
# yticks = ax.set_yticks(fontsize=20)

ax.set_ylabel("${}$".format(theory_name), fontsize=20);
plt.tight_layout();

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.savefig("../figures/real_gan_vs_theory_pair_kappa_{}.pdf".format(varname))


# In[ ]:


# Some experiments varying $p$ and $q$ in the RMT calculations.

# In[ ]:


def parse_name(name):
    a, b = name.split("q")
    q = int(b)
    p = int(a[1:])
    return f"p={p}, q={q}"

fig = plt.figure(figsize=multiply_figsize(figsize, (1, 1)))
plt.subplot(1, 1, 1);
styles = ['-', '-.', ':',  ] * 4
markers = ['o', 's', 'd', '*']*2
for style, marker, (exp, results) in zip(styles, markers, vary_pq_kappa_results.items()):
#     if exp == "p3q3":
#         continue
    min_uDs, _ = results
    plot_reals = discrims
    plot_rmts = min_uDs[kappas >= min_kappa]
    varname= "L_D"
    theory_name = "\\theta_D"

    plt.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, label=parse_name(exp), linestyle=style);
plt.xlabel("$\kappa$", fontsize=20);
plt.xscale('log')
xticks = plt.xticks(fontsize=10)
yticks = plt.yticks(fontsize=10)
plt.ylabel("${}$".format(theory_name), fontsize=20);
plt.legend();

plt.ylabel("min ${}$".format(varname), fontsize=20);

plt.tight_layout();

plt.savefig("../figures/gan_theory_kappa_{}_vary_pq.pdf".format(varname))


# In[ ]:


fig, axs= plt.subplots(1, len(vary_pq_kappa_results), figsize=multiply_figsize(figsize, (len(vary_pq_kappa_results), 1)))

for ind, (exp, results) in enumerate(vary_pq_kappa_results.items()):
    min_uDs, _ = results
    plot_reals = discrims
    plot_reals = pd.Series(plot_reals).rolling(5).mean().backfill().values
    plot_reals = plot_reals[kappas >= min_kappa]
    plot_rmts = min_uDs[kappas >= min_kappa]

    varname= "L_D"
    theory_name = "\\theta_D"
    ax1 = axs[ind]
    # plt.subplot(1, 2, 1);
    ln1=ax1.plot(kappas[kappas >= min_kappa], plot_reals, 'x--', markersize=10, label="$L_D$");
    ax1.set_xlabel("$\kappa$", fontsize=20);
    ax1.set_ylabel("min ${}$".format(varname), fontsize=20);

    ax = ax1.twinx()
    # plt.subplot(1, 2, 2);
    ln2=ax.plot(kappas[kappas >= min_kappa], plot_rmts, linewidth=3, color="tab:red", label="$\\theta_D$");

    plt.xlabel("$\kappa$", fontsize=20);
    plt.xscale('log')
    xticks = plt.xticks(fontsize=20)
    # yticks = ax.set_yticks(fontsize=20)

    ax.set_ylabel("${}$".format(theory_name), fontsize=20);

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.set_title(parse_name(exp))
plt.tight_layout();
plt.savefig("../figures/real_gan_vs_theory_kappa_pair_variety{}.pdf".format(varname))


# In[ ]:


# def get_loss(ind=None, kappa=None):
#     if ind is not None:
#         kappa = sorted(list(results.keys()))[ind]
#     discrims = pd.DataFrame(np.array([r[0] for r in results[kappa]]).T).values
#     gens = pd.DataFrame(np.array([r[1] for r in results[kappa]]).T).values
#     return discrims.T, gens.T


# In[ ]:


# plot_kappas = kappas[::11]
# for plot_kappa in plot_kappas:
#     kappa = kappas[np.argmin(np.abs(kappas - plot_kappa))]
#     discrim_losses, gen_losses = get_loss(kappa=kappa)
#     discrim_failure = max(zip(discrim_losses, gen_losses), key=lambda x: sum((x[0] - x[1])[-700:]))
# #     gen_failure = max(zip(discrim_losses, gen_losses), key=lambda x: sum((x[1] - x[0])[-200:]))
#     success = min(zip(discrim_losses, gen_losses), key=lambda x: np.abs(x[0][-200:]-x[1][-200:]).sum())
#     plt.figure(figsize=(11,3))
#     for i, traces in enumerate([discrim_failure, gen_failure, success]):
#         plt.subplot(1, 3, i+1)
#         plt.plot(traces[0], label="discrimnator");
#         plt.plot(traces[1], alpha=0.5, label="generator");
#         plt.xlabel("training iterations");
#         plt.ylabel("loss");
#         plt.legend();
#     plt.suptitle("$\kappa={:.7f}$".format(kappa));


# In[ ]:
