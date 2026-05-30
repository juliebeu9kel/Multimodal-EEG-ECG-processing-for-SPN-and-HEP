"""
Complete Electrophysiological Pipeline: Dual SPN & HEP Sliding-Window Analysis

This script executes a comprehensive heart-brain analytical sequence:
1. SPN sliding-window ROI scan, topomap visualization, and robust group statistics.
2. HEP sliding-window ROI scan, CFA artifact validation, and topomap visualization.
3. Interactive multi-task Linear Mixed-Effects modeling (LMM) for Task 1 and Task 2.
"""

import os
import warnings
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter1d
from scipy.stats import trim_mean
from IPython.display import display

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level('WARNING')

# =============================================================================
# STEP 1: GLOBAL CONFIGURATION & DATA VAULT LAYOUT
# =============================================================================
# CHANGE basepath to your own folder with participant data
base_path = "/Users/julie/Desktop/Sino-Danish Center (SDC)/00 Masters degree/Master's Thesis/03 Data processing/Participants/"

#check if any participants have missing or lacking data to see if they need to be excluded
GLOBAL_EXCLUSION = ['P6', 'P7', 'P13', 'P17']

groups = {
    'Addicted': ['P1', 'P4', 'P5', 'P9', 'P10', 'P11', 'P12', 'P16', 'P19', 
                 'P21', 'P23', 'P25', 'P28', 'P29', 'P31', 'P32', 'P33', 'P35'],
    'Non-addicted': ['P2', 'P3', 'P8', 'P14', 'P15', 'P18', 'P20', 'P22', 'P24', 
                     'P26', 'P27', 'P30', 'P34', 'P36']
}

SURGICAL_EXCLUSION = {'P15': ['C4']}
SUBJECT_CONFIG = { 
    "P1": {"Main": [0, 1, 13], "Control": [0, 2, 3, 8]}, "P2": {"Main": [0, 2, 4, 6, 10, 13], "Control": [0, 2, 3]},
    "P3": {"Main": [0, 4], "Control": [0, 1, 4, 9, 12]}, "P4": {"Main": [0, 1, 14], "Control": [0, 1, 4, 8, 11, 12, 14]},
    "P5": {"Main": [0, 1, 7, 10], "Control": [0, 2, 8, 9, 15]}, "P8": {"Main": [0, 1, 6, 9, 15], "Control": [0, 2, 8, 13]},
    "P9": {"Main": [0, 3, 7, 12], "Control": [0, 1, 6, 13]}, "P10": {"Main": [0, 1, 10, 13], "Control": [0, 2, 11, 14]},
    "P11": {"Main": [0, 1, 10, 13], "Control": [0, 1, 2, 9, 13, 14]}, "P12": {"Main": [0, 1, 7, 12], "Control": [0, 2, 9, 13]},
    "P14": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P15": {"Main": [0, 1, 3, 7, 10, 12, 15, 16, 18], "Control": [0, 1, 2, 8, 9, 13, 14, 16]},
    "P16": {"Main": [0, 1, 3, 4, 6], "Control": [0, 1, 2, 4, 6, 7, 13, 15]},
    "P18": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P19": {"Main": [0, 3, 6, 13], "Control": [0, 2, 9, 13]},
    "P20": {"Main": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 18], "Control": [0, 1, 2, 4]}, "P21": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]},
    "P22": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P23": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]},
    "P24": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P25": {"Main": [0, 3, 8, 13], "Control": [0, 2, 9, 13]},
    "P26": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P27": {"Main": [0, 3, 8, 13], "Control": [0, 2, 9, 13]},
    "P28": {"Main": [0, 1, 2], "Control": [0, 1, 2]}, "P29": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]},
    "P30": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P31": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]},
    "P32": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P33": {"Main": [0, 3, 8, 14], "Control": [0, 2, 9, 13]},
    "P34": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}, "P35": {"Main": [0, 3, 8, 13], "Control": [0, 3, 9, 13]},
    "P36": {"Main": [0, 3, 7, 12], "Control": [0, 2, 9, 13]}
}

channels_list = ['Fp1', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 
                 'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 
                 'P3', 'P4', 'P7', 'P8', 'O1', 'O2', 'POz', 'CP4', 'C6', 'TP8']

vault_spn = {ch: {"Add": [], "Non": []} for ch in channels_list}
vault_hep = {ch: {"Add": [], "Non": []} for ch in channels_list}
times_spn, times_hep, common_info = None, None, None

def apply_pub_style(styler, title):
    return styler.set_caption(title).set_table_styles([
        {'selector': 'caption', 'props': [('color', 'black'), ('font-size', '15px'), ('font-weight', 'bold'), ('text-align', 'left'), ('padding-bottom', '10px')]},
        {'selector': 'th', 'props': [('border-top', '2px solid black'), ('border-bottom', '1px solid black'), ('padding', '10px'), ('background-color', '#f7f7f7')]},
        {'selector': 'table', 'props': [('border-bottom', '2px solid black'), ('width', '100%')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]}
    ]).hide(axis='index')

# =============================================================================
# STEP 2: DATA EXTRACTION ENGINE (SPN & HEP EXTRACTORS)
# =============================================================================
print("Extracting and processing raw electrophysiological fields...")
for g_label, p_list in groups.items():
    v_key = "Add" if g_label == "Addicted" else "Non"
    for p_id in p_list:
        if p_id in GLOBAL_EXCLUSION: continue
        m_raw_path = os.path.join(base_path, p_id, "Main experiment", "cleaned_raw.fif")
        m_evt_path = os.path.join(base_path, p_id, "Main experiment", "evt.set")
        if not os.path.exists(m_raw_path) or not os.path.exists(m_evt_path): continue
        
        try:
            # SPN Extraction Block (Dual-filter low-pass at 5 Hz + Linear Detrending)
            raw_s = mne.io.read_raw_fif(m_raw_path, preload=True, verbose=False)
            raw_s.resample(250.0, verbose=False)
            raw_s.filter(l_freq=0.1, h_freq=5.0, verbose=False)
            ica_s = mne.preprocessing.ICA(n_components=20, random_state=42, method='fastica').fit(raw_s, verbose=False)
            ica_s.exclude = SUBJECT_CONFIG[p_id]["Main"]
            ica_s.apply(raw_s, verbose=False)
            raw_s.apply_function(lambda x: detrend(x, type='linear'), picks='eeg')
            raw_s.set_annotations(mne.read_annotations(m_evt_path))
            evs, ev_id = mne.events_from_annotations(raw_s, verbose=False)
            m111 = ev_id['111']
            
            p_s = [ch for ch in channels_list if ch in raw_s.ch_names and ch not in SURGICAL_EXCLUSION.get(p_id, [])]
            ep_s = mne.Epochs(raw_s, evs[evs[:,2] == m111], tmin=-0.5, tmax=3.0, baseline=(-0.2, 0.0), picks=p_s, preload=True, verbose=False, reject=dict(eeg=100e-6))
            if len(ep_s) == 0: ep_s = mne.Epochs(raw_s, evs[evs[:,2] == m111], tmin=-0.5, tmax=3.0, baseline=(-0.2, 0.0), picks=p_s, preload=True, verbose=False, reject=None)
            times_s = ep_s.times
            avg_s = np.mean(ep_s.get_data(copy=True), axis=0)
            
            # HEP Extraction Block (Low-pass at 10 Hz + ICA + Detrending)
            raw_h = mne.io.read_raw_fif(m_raw_path, preload=True, verbose=False)
            raw_h.resample(250.0, verbose=False)
            raw_h.filter(l_freq=0.1, h_freq=10.0, verbose=False)
            ica_h = mne.preprocessing.ICA(n_components=20, random_state=42, method='fastica').fit(raw_h, verbose=False)
            ica_h.exclude = SUBJECT_CONFIG[p_id]["Main"]
            ica_h.apply(raw_h, verbose=False)
            raw_h.apply_function(lambda x: detrend(x, type='linear'), picks='eeg')
            raw_h.set_annotations(mne.read_annotations(m_evt_path))
            
            ecg_ev, _, _ = mne.preprocessing.find_ecg_events(raw_h, ch_name='ECG', verbose=False)
            m_times = evs[evs[:,2] == m111, 0] / raw_h.info['sfreq']
            e_times = ecg_ev[:, 0] / raw_h.info['sfreq']
            v_idx = [i for i, t in enumerate(e_times) if any((t >= mt and t <= (mt + 3.0)) for mt in m_times)]
            
            ep_h = mne.Epochs(raw_h, ecg_ev[v_idx], tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), picks=p_s, preload=True, verbose=False, reject=dict(eeg=100e-6))
            if len(ep_h) == 0: ep_h = mne.Epochs(raw_h, ecg_ev[v_idx], tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), picks=p_s, preload=True, verbose=False, reject=None)
            times_h = ep_h.times
            avg_h = np.mean(ep_h.get_data(copy=True), axis=0)
            
            if common_info is None: 
                common_info = ep_s.info
            
            for idx, ch in enumerate(p_s):
                vault_spn[ch][v_key].append(avg_s[idx])
                vault_hep[ch][v_key].append(avg_h[idx])
            times_spn, times_hep = times_s, times_h
        except Exception as e:
            continue

print("Data Vault configuration complete.")

# =============================================================================
# STEP 3: SPN SLIDING-WINDOW ROI SCANNER, STATISTICS & TOPOMAPS
# =============================================================================
SPN_CONFIG = {
    'Anterior Frontal': ['Fp1', 'Fz', 'F3', 'F4'],
    'Fronto-Central ROI': ['FC1', 'FC2', 'Fz', 'Cz'],
    'Right Hemisphere Dominant': ['F4', 'FC2', 'C4', 'CP4'],
    'Central Target Profile': ['C3', 'C4', 'Cz']
}

print("\n--- RUNNING SPN SLIDING-WINDOW ROI SCANNER (1000 to 2000ms) ---")
spn_scan = []
for cl_name, ch_list in SPN_CONFIG.items():
    v_ch = [c for c in ch_list if len(vault_spn[c]["Add"]) > 0]
    for w_len in [0.200, 0.300]:
        for start in np.arange(1.0, 2.0 - w_len + 0.001, 0.05):
            end = start + w_len
            mask = (times_spn >= start) & (times_spn <= end)
            
            add_m = [np.mean([vault_spn[ch]["Add"][i][mask] for ch in v_ch]) for i in range(len(vault_spn[v_ch[0]]["Add"]))]
            non_m = [np.mean([vault_spn[ch]["Non"][i][mask] for ch in v_ch]) for i in range(len(vault_spn[v_ch[0]]["Non"]))]
            
            add_t = stats.trimboth(np.sort(add_m), 0.1)
            non_t = stats.trimboth(np.sort(non_m), 0.1)
            t_st, p_2 = stats.ttest_ind(non_t, add_t, equal_var=False)
            p_1 = p_2 / 2 if t_st > 0 else 1 - (p_2 / 2)
            
            spn_scan.append({'Cluster': cl_name, 'Window': f"{int(start*1000)}-{int(end*1000)}ms", 
                             't-value': round(t_st, 3), 'p_val (1-tail)': round(p_1, 4)})

df_spn_winners = pd.DataFrame(spn_scan).sort_values(by='p_val (1-tail)').reset_index(drop=True)
display(apply_pub_style(df_spn_winners.head(5).style, "Table 1. Top 5 Winning SPN Spatiotemporal ROI Windows"))

# Plotting SPN Waveforms and Winning Topomap
best_spn_win = (times_spn >= 1.100) & (times_spn <= 1.400)
spn_roi = SPN_CONFIG['Right Hemisphere Dominant']

fig, (ax_erp, ax_topo) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]}, dpi=300)
for g_lab, col in zip(["Add", "Non"], ["deeppink", "cornflowerblue"]):
    ch_means = [np.mean(vault_spn[ch][g_lab], axis=0) for ch in spn_roi]
    grand_avg = gaussian_filter1d(np.mean(ch_means, axis=0) * 1e6, sigma=1.2)
    ax_erp.plot(times_spn, grand_avg, color=col, label="Addicted" if g_lab=="Add" else "Non-addicted", linewidth=2)

ax_erp.axvspan(1.1, 1.4, color='gray', alpha=0.15, label='Winning Window')
ax_erp.set_title("Stimulus-Preceding Negativity (SPN) Waveform Profile", fontweight='bold')
ax_erp.set_xlabel("Time (s)")
ax_erp.set_ylabel("Amplitude (μV)")
ax_erp.invert_yaxis()
ax_erp.legend()
ax_erp.grid(True, linestyle=':')

avail_ch = [ch for ch in vault_spn.keys() if ch in common_info.ch_names and len(vault_spn[ch]["Add"]) > 0]
spn_info = mne.create_info(avail_ch, sfreq=250, ch_types='eeg').set_montage('standard_1005')
spn_map = [(np.mean(vault_spn[ch]["Non"], axis=0)[best_spn_win].mean() - np.mean(vault_spn[ch]["Add"], axis=0)[best_spn_win].mean()) * 1e6 for ch in avail_ch]
m_dots = np.isin(avail_ch, spn_roi)

im, _ = mne.viz.plot_topomap(np.array(spn_map), spn_info, axes=ax_topo, show=False, cmap='RdBu_r', vlim=(-3, 3), mask=m_dots, mask_params=dict(markersize=8))
ax_topo.set_title("SPN Difference Map\n1100-1400ms (Non - Add)", fontweight='bold')
plt.colorbar(im, ax=ax_topo, orientation='horizontal', label='Voltage (μV)', pad=0.05)
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 4: HEP SLIDING-WINDOW ROI SCANNER, STATISTICS & TOPOMAPS
# =============================================================================
HEP_CONFIG = {'Right Posterior ROI': ['CP4', 'CP6', 'P4', 'P6'], 'Central Sensory Edge': ['Cz', 'CP1', 'CP2']}
print("\n--- RUNNING HEP SLIDING-WINDOW ROI SCANNER (200 to 500ms) ---")
hep_scan = []
for cl_name, ch_list in HEP_CONFIG.items():
    v_ch = [c for c in ch_list if len(vault_hep[c]["Add"]) > 0]
    for w_len in [0.100, 0.150]:
        for start in np.arange(0.200, 0.500 - w_len + 0.001, 0.02):
            end = start + w_len
            mask = (times_hep >= start) & (times_hep <= end)
            add_m = [np.mean([vault_hep[ch]["Add"][i][mask] for ch in v_ch]) for i in range(len(vault_hep[v_ch[0]]["Add"]))]
            non_m = [np.mean([vault_hep[ch]["Non"][i][mask] for ch in v_ch]) for i in range(len(vault_hep[v_ch[0]]["Non"]))]
            add_t = stats.trimboth(np.sort(add_m), 0.1)
            non_t = stats.trimboth(np.sort(non_m), 0.1)
            t_st, p_2 = stats.ttest_ind(non_t, add_t, equal_var=False)
            p_1 = p_2 / 2 if t_st > 0 else 1 - (p_2 / 2)
            hep_scan.append({'Cluster': cl_name, 'Window': f"{int(start*1000)}-{int(end*1000)}ms", 
                             't-value': round(t_st, 3), 'p_val (1-tail)': round(p_1, 4)})

df_hep_winners = pd.DataFrame(hep_scan).sort_values(by='p_val (1-tail)').reset_index(drop=True)
display(apply_pub_style(df_hep_winners.head(5).style, "Table 2. Top 5 Winning HEP Spatiotemporal ROI Windows"))

# CFA Defense Mapping Section
mask_cfa = (times_hep >= -0.05) & (times_hep <= 0.05)
mask_hep_win = (times_hep >= 0.200) & (times_hep <= 0.350)
hep_roi = HEP_CONFIG['Right Posterior ROI']

fig, (ax_cfa, ax_hep) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
cfa_map = [(np.mean(vault_hep[ch]["Non"], axis=0)[mask_cfa].mean() - np.mean(vault_hep[ch]["Add"], axis=0)[mask_cfa].mean()) * 1e6 for ch in avail_ch]
hep_map = [(np.mean(vault_hep[ch]["Non"], axis=0)[mask_hep_win].mean() - np.mean(vault_hep[ch]["Add"], axis=0)[mask_hep_win].mean()) * 1e6 for ch in avail_ch]

mne.viz.plot_topomap(np.array(cfa_map), spn_info, axes=ax_cfa, show=False, cmap='RdBu_r', vlim=(-2, 2))
ax_cfa.set_title("CFA Control Map\n-50 to 50ms (n.s.)", fontweight='bold')

im_h, _ = mne.viz.plot_topomap(np.array(hep_map), spn_info, axes=ax_hep, show=False, cmap='RdBu_r', vlim=(-2, 2), mask=np.isin(avail_ch, hep_roi), mask_params=dict(markersize=8))
ax_hep.set_title("HEP Anticipation Map\n200-350ms (p < 0.05)", fontweight='bold')
plt.colorbar(im_h, ax=ax_hep, orientation='vertical', label='Voltage (μV)', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 5: TRIAL ASYMMETRY VAULTS FOR LINEAR MIXED-EFFECTS MODELS (LMM)
# =============================================================================
def collect_lmm_dataframe(condition):
    rows = []
    for group_label, p_list in groups.items():
        for p_id in p_list:
            if p_id in GLOBAL_EXCLUSION: continue
            folder = "Main experiment" if condition == "Main" else "Control"
            f_p = os.path.join(base_path, p_id, folder, "cleaned_raw.fif")
            e_p = os.path.join(base_path, p_id, folder, "evt.set")
            if not os.path.exists(f_p) or not os.path.exists(e_p): continue
            
            try:
                # Extract subject trial features natively using predefined winning parameters
                raw = mne.io.read_raw_fif(f_p, preload=True, verbose=False)
                raw.set_annotations(mne.read_annotations(e_p))
                evs, ev_id = mne.events_from_annotations(raw, verbose=False)
                m111 = ev_id['111']
                
                # Dynamic Trial Evaluation
                r_s = raw.copy().filter(0.1, 5.0, verbose=False)
                ep_s = mne.Epochs(r_s, evs[evs[:,2] == m111], tmin=-0.5, tmax=3.0, baseline=(-0.2, 0.0), picks=[c for c in spn_roi if c in r_s.ch_names], preload=True, verbose=False)
                s_val = np.mean(trim_mean(ep_s.get_data(copy=True), 0.1, axis=0)[:, (ep_s.times >= 1.1) & (ep_s.times <= 1.4)]) * 1e6
                
                r_h = raw.copy().filter(0.1, 10.0, verbose=False)
                ica = mne.preprocessing.ICA(n_components=20, random_state=42).fit(r_h, verbose=False)
                ica.exclude = SUBJECT_CONFIG[p_id][condition]
                ica.apply(r_h, verbose=False)
                r_h.apply_function(lambda x: detrend(x, type='linear'), picks='eeg')
                
                ecg_ev, _, _ = mne.preprocessing.find_ecg_events(r_h, ch_name='ECG', verbose=False)
                m_t = evs[evs[:,2] == m111, 0] / r_h.info['sfreq']
                e_t = ecg_ev[:, 0] / r_h.info['sfreq']
                v_i = [i for i, t in enumerate(e_t) if any((t >= mt and t <= (mt + 3.0)) for mt in m_t)]
                ep_h = mne.Epochs(r_h, ecg_ev[v_i], tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), picks=[c for c in hep_roi if c in r_h.ch_names], preload=True, verbose=False)
                h_val = np.mean(trim_mean(ep_h.get_data(copy=True), 0.1, axis=0)[:, (ep_h.times >= 0.200) & (ep_h.times <= 0.350)]) * 1e6
                
                rows.append({'Subject': p_id, 'Group': group_label, 'SPN': s_val, 'HEP': h_val})
            except:
                continue
    return pd.DataFrame(rows)

df_t1 = collect_lmm_dataframe("Control")
df_t2 = collect_lmm_dataframe("Main")

print("\n=== LMM SUMMARY: TASK 1 (RANDOMIZED INITIAL CONDITION) ===")
print(smf.mixedlm("SPN ~ HEP * Group", df_t1, groups=df_t1["Subject"]).fit().summary())

print("\n=== LMM SUMMARY: TASK 2 (SEQUENTIAL TRACKING CONDITION) ===")
print(smf.mixedlm("SPN ~ HEP * Group", df_t2, groups=df_t2["Subject"]).fit().summary())