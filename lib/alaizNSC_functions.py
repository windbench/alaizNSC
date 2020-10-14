import pandas as pd
import numpy as np
import datetime
import netCDF4
import utm
import xarray as xr
import math
import rasterio as rio
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
import seaborn as sns
from scipy import stats
from yaml import safe_load

from lib.variables_dictionary.variables import Variables
from lib.variables_dictionary.variables import nc_global_attributes_from_yaml

def A_vs_RS_scatter(sector, var, hue_var, masts_obs, iSC1, iSC2):
    WD_ref = masts_obs['MP5']['DVM118']
    if sector == 'N':
        isector = WD_ref[(WD_ref >= 355) | (WD_ref <= 25)].index
    elif sector == 'S':
        isector = WD_ref[(WD_ref >= 145) & (WD_ref <= 195)].index

    isectorSC1 = isector.intersection(iSC1)
    SC1_sector = masts_obs['MP5'].loc[isectorSC1]['VM118P'].rename('U_ref')
    SC1_sector = pd.concat([SC1_sector, masts_obs['MP5'].loc[isectorSC1]['Fr_81_97'].rename('Fr_ref')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['MP5'].loc[isectorSC1]['alpha'].rename('alpha_ref')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['A4'].loc[isectorSC1]['VMP118'].rename('U_A4')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['A5'].loc[isectorSC1]['VMP118'].rename('U_A5')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['A6'].loc[isectorSC1]['VMP118'].rename('U_A6')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['A4'].loc[isectorSC1]['alpha'].rename('alpha_A4')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['A5'].loc[isectorSC1]['alpha'].rename('alpha_A5')], axis=1)
    SC1_sector = pd.concat([SC1_sector, masts_obs['A6'].loc[isectorSC1]['alpha'].rename('alpha_A6')], axis=1)

    isectorSC2 = isector.intersection(iSC2)
    SC2_sector = masts_obs['MP5'].loc[isectorSC2]['VM118P'].rename('U_ref')
    SC2_sector = pd.concat([SC2_sector, masts_obs['MP5'].loc[isectorSC2]['Fr_81_97'].rename('Fr_ref')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['MP5'].loc[isectorSC2]['alpha'].rename('alpha_ref')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['A1'].loc[isectorSC2]['VMP118'].rename('U_A1')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['A2'].loc[isectorSC2]['VMP118'].rename('U_A2')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['A3'].loc[isectorSC2]['VMP118'].rename('U_A3')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['A1'].loc[isectorSC2]['alpha'].rename('alpha_A1')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['A2'].loc[isectorSC2]['alpha'].rename('alpha_A2')], axis=1)
    SC2_sector = pd.concat([SC2_sector, masts_obs['A3'].loc[isectorSC2]['alpha'].rename('alpha_A3')], axis=1)
    
    fig, ax = plt.subplots(2, 3, figsize = (15,10))
    sns.scatterplot(data=SC2_sector, x=var+'_ref', y=var+'_A1', hue=hue_var, palette = 'coolwarm_r', ax = ax[0][0], legend = False, zorder = 2)
    sns.regplot(data=SC2_sector, x=var+'_ref', y=var+'_A1', ci = 95, fit_reg = 'True', marker = '', color = 'grey', ax=ax[0][0])
    sns.scatterplot(data=SC2_sector, x=var+'_ref', y=var+'_A2', hue=hue_var, palette = 'coolwarm_r', ax = ax[0][1], legend = False, zorder = 2)
    sns.regplot(data=SC2_sector, x=var+'_ref', y=var+'_A2', ci = 95, marker = '', color = 'grey', ax=ax[0][1])
    g = sns.scatterplot(data=SC2_sector, x=var+'_ref', y=var+'_A3', hue=hue_var, palette = 'coolwarm_r', ax = ax[0][2], zorder = 2)
    sns.regplot(data=SC2_sector, x=var+'_ref', y=var+'_A3', ci = 95, marker = '', color = 'grey', ax=ax[0][2])
    sns.scatterplot(data=SC1_sector, x=var+'_ref', y=var+'_A4', hue=hue_var, palette = 'coolwarm_r', ax = ax[1][0], legend = False, zorder = 2)
    sns.regplot(data=SC1_sector, x=var+'_ref', y=var+'_A4', ci = 95, marker = '', color = 'grey', ax=ax[1][0])
    sns.scatterplot(data=SC1_sector, x=var+'_ref', y=var+'_A5', hue=hue_var, palette = 'coolwarm_r', ax = ax[1][1], legend = False, zorder = 2)
    sns.regplot(data=SC1_sector, x=var+'_ref', y=var+'_A5', ci = 95, marker = '', color = 'grey', ax=ax[1][1])
    sns.scatterplot(data=SC1_sector, x=var+'_ref', y=var+'_A6', hue=hue_var, palette = 'coolwarm_r', ax = ax[1][2], legend = False, zorder = 2)
    sns.regplot(data=SC1_sector, x=var+'_ref', y=var+'_A6', ci = 95, marker = '', color = 'grey', ax=ax[1][2])
    g.legend(loc=2,bbox_to_anchor=(1.05, 1))
    ax[0][0].set_title(sector + ' sector: A1 vs RS'); ax[0][1].set_title(sector + ' sector: A2 vs RS'); 
    ax[0][2].set_title(sector + ' sector: A3 vs RS'); ax[1][0].set_title(sector + ' sector: A4 vs RS'); 
    ax[1][1].set_title(sector + ' sector: A5 vs RS'); ax[1][2].set_title(sector + ' sector: A6 vs RS');
    if var == 'alpha':
        ax[0][0].set_xlabel(r'$\alpha_{ref}$'); ax[0][1].set_xlabel(r'$\alpha_{ref}$'); ax[0][2].set_xlabel(r'$\alpha_{ref}$')
        ax[1][0].set_xlabel(r'$\alpha_{ref}$'); ax[1][1].set_xlabel(r'$\alpha_{ref}$'); ax[1][2].set_xlabel(r'$\alpha_{ref}$')
        ax[0][0].set_ylabel(r'$\alpha_{A1}$'); ax[0][1].set_ylabel(r'$\alpha_{A2}$'); ax[0][2].set_ylabel(r'$\alpha_{A3}$')
        ax[1][0].set_ylabel(r'$\alpha_{A4}$'); ax[1][1].set_ylabel(r'$\alpha_{A5}$'); ax[1][2].set_ylabel(r'$\alpha_{A6}$')
        ax[0][0].set_xlim([-0.5,1]); ax[0][1].set_xlim([-0.5,1]); ax[0][2].set_xlim([-0.5,1])
        ax[1][0].set_xlim([-0.5,1]); ax[1][1].set_xlim([-0.5,1]); ax[1][2].set_xlim([-0.5,1])
        ax[0][0].set_ylim([-0.5,1]); ax[0][1].set_ylim([-0.5,1]); ax[0][2].set_ylim([-0.5,1])
        ax[1][0].set_ylim([-0.5,1]); ax[1][1].set_ylim([-0.5,1]); ax[1][2].set_ylim([-0.5,1])
    elif var == 'U':
        ax[0][0].set_xlabel(r'$U_{ref}$'); ax[0][1].set_xlabel(r'$U_{ref}$'); ax[0][2].set_xlabel(r'$U_{ref}$')
        ax[1][0].set_xlabel(r'$U_{ref}$'); ax[1][1].set_xlabel(r'$U_{ref}$'); ax[1][2].set_xlabel(r'$U_{ref}$')
        ax[0][0].set_ylabel(r'$U_{A1}$'); ax[0][1].set_ylabel(r'$U_{A2}$'); ax[0][2].set_ylabel(r'$U_{A3}$')
        ax[1][0].set_ylabel(r'$U_{A4}$'); ax[1][1].set_ylabel(r'$U_{A5}$'); ax[1][2].set_ylabel(r'$U_{A6}$')
        ax[0][0].set_xlim([4,16]); ax[0][1].set_xlim([4,16]); ax[0][2].set_xlim([4,16])
        ax[1][0].set_xlim([4,16]); ax[1][1].set_xlim([4,16]); ax[1][2].set_xlim([4,16])
        ax[0][0].set_ylim([4,16]); ax[0][1].set_ylim([4,16]); ax[0][2].set_ylim([4,16])
        ax[1][0].set_ylim([4,16]); ax[1][1].set_ylim([4,16]); ax[1][2].set_ylim([4,16])        
    ax[0][0].grid(zorder = 0); ax[0][1].grid(zorder = 0); ax[0][2].grid(zorder = 0);
    ax[1][0].grid(zorder = 0); ax[1][1].grid(zorder = 0); ax[1][2].grid(zorder = 0);

def ref_wind_conditions_diurnal_cycle(period,masts_obs):
    WD_ref = masts_obs['MP5']['DVM118']
    iN = WD_ref[(WD_ref >= 355) | (WD_ref <= 25)].index
    iN = period.intersection(iN)
    iS = WD_ref[(WD_ref >= 145) & (WD_ref <= 195)].index
    iS = period.intersection(iS)

    fig, ax = plt.subplots(2, 4, figsize = (22,8), sharex = True)

    sns.lineplot(x = 'tod', y = 'VM118P', data=masts_obs['MP5'].loc[iN], ci = 'sd', ax = ax[0][0])
    g = sns.lineplot(x = 'tod', y = 'Fr_81_97', data=masts_obs['MP5'].loc[iN], ci = 'sd', ax = ax[0][1], label = 'Fr(81/97)')
    g = sns.lineplot(x = 'tod', y = 'Fr_2_40', data=masts_obs['MP5'].loc[iN], ci = 'sd', ax = ax[0][1], label = 'Fr(2/40)')
    sns.lineplot(x = 'tod', y = 'TI118', data=masts_obs['MP5'].loc[iN], ci = 'sd', ax = ax[0][2])
    sns.lineplot(x = 'tod', y = 'alpha', data=masts_obs['MP5'].loc[iN], ci = 'sd', ax = ax[0][3])
    ax[0][1].hlines(y=[-0.25,0.25], xmin = 0, xmax = 24, color='k', linestyle=':')

    ax[0][0].set_xlim([0,24]); ax[0][0].set_xticks(np.arange(0,26,2))
    ax[0][0].set_ylim([5,15]); ax[0][0].set_ylabel(r'$U_{ref}$'); ax[0][0].grid(); ax[0][0].set_title('N sector: Wind speed')
    ax[0][1].set_ylim([-3,2]); ax[0][1].set_ylabel(r'$Fr_{ref}^{-1}$'); ax[0][1].grid(); ax[0][1].set_title('N sector: Stability')
    ax[0][2].set_ylim([0.02,0.17]); ax[0][2].set_ylabel(r'$TI_{ref}$'); ax[0][2].grid(); ax[0][2].set_title('N sector: Turbulence intensity')
    ax[0][3].set_ylim([-0.2,0.4]); ax[0][3].set_ylabel(r'$\alpha_{ref}$'); ax[0][3].grid(); ax[0][3].set_title('N sector:  Wind shear')
    g.legend(loc='lower right'); 
    
    sns.lineplot(x = 'tod', y = 'VM118P', data=masts_obs['MP5'].loc[iS], ci = 'sd', ax = ax[1][0])
    g = sns.lineplot(x = 'tod', y = 'Fr_81_97', data=masts_obs['MP5'].loc[iS], ci = 'sd', ax = ax[1][1], label = 'Fr(81/97)')
    g = sns.lineplot(x = 'tod', y = 'Fr_2_40', data=masts_obs['MP5'].loc[iS], ci = 'sd', ax = ax[1][1], label = 'Fr(2/40)')
    sns.lineplot(x = 'tod', y = 'TI118', data=masts_obs['MP5'].loc[iS], ci = 'sd', ax = ax[1][2])
    sns.lineplot(x = 'tod', y = 'alpha', data=masts_obs['MP5'].loc[iS], ci = 'sd', ax = ax[1][3])
    ax[1][1].hlines(y=[-0.25,0.25], xmin = 0, xmax = 24, color='k', linestyle=':')

    ax[1][0].set_xlim([0,24]); ax[1][0].set_xticks(np.arange(0,26,2))
    ax[1][0].set_ylim([5,15]); ax[1][0].set_ylabel(r'$U_{ref}$'); ax[1][0].grid(); ax[1][0].set_xlabel('time of day'); ax[1][0].set_title('S sector: Wind speed')
    ax[1][1].set_ylim([-3,2]); ax[1][1].set_ylabel(r'$Fr_{ref}^{-1}$'); ax[1][1].grid(); ax[1][1].set_xlabel('time of day'); ax[1][1].set_title('S sector: Stability')
    ax[1][2].set_ylim([0.02,0.17]); ax[1][2].set_ylabel(r'$TI_{ref}$'); ax[1][2].grid(); ax[1][2].set_xlabel('time of day'); ax[1][2].set_title('S sector: Turbulence intensity')
    ax[1][3].set_ylim([-0.2,0.4]); ax[1][3].set_ylabel(r'$\alpha_{ref}$'); ax[1][3].grid(); ax[1][3].set_xlabel('time of day')
    g.legend(loc='lower right'); ax[1][3].set_title('S sector: Wind shear')

def ref_wind_conditions_hist(period,masts_obs):
    """
    Plot histograms of wind conditions for North and South sector
    """
    mast = 'MP5'
    WD_ref = masts_obs['MP5'].loc[period]['DVM118']
    iN = WD_ref[(WD_ref > 345) | (WD_ref < 15)].index
    iS = WD_ref[(WD_ref > 165) & (WD_ref < 195)].index
    fig, ax = plt.subplots(2, 3, figsize = (16,8))
    masts_obs[mast].loc[iN,'VM118P'].hist(bins = np.arange(0,24,1), ax = ax[0][0], color = 'silver') 
    masts_obs[mast].loc[iN,'TI118'].hist(bins = np.arange(0,0.3,0.01), ax = ax[0][1], color = 'silver')
    masts_obs[mast].loc[iN,'alpha'].hist(bins = np.arange(-0.6,1,0.02), ax = ax[0][2], color = 'silver')
    masts_obs[mast].loc[iS,'VM118P'].hist(bins = np.arange(0,24,1), ax = ax[1][0], color = 'silver')
    masts_obs[mast].loc[iS,'TI118'].hist(bins = np.arange(0,0.3,0.01), ax = ax[1][1], color = 'silver')
    masts_obs[mast].loc[iS,'alpha'].hist(bins = np.arange(-0.6,1,0.02), ax = ax[1][2], color = 'silver')
    ax[0][0].set_title(mast + ': Wind speed at a hub-height of 118 m')
    ax[0][1].set_title(mast + ': Turbulence Intensity at 118 m')
    ax[0][2].set_title(mast + r': Wind shear ($\alpha$) between 78 and 118 m')
    ax[0][0].set_ylabel('North Sector')
    ax[1][0].set_ylabel('South Sector')
    ax[1][0].set_xlabel(r'$U_{ref}$ [$m s^{-1}$]')
    ax[1][1].set_xlabel(r'$TI_{ref}$')
    ax[1][2].set_xlabel(r'$\alpha_{ref}$');


def bin_hist(sector, stab, WDbins_label, Frbins_label, binmap_SC1, binmap_SC2, binmap_MP5, U_ref, WD_ref, Fr_ref, Sbins, WDbins, Frbins):
    """
    Compare short-term with long-term bin histograms
    """
    isector = np.where(WDbins_label == sector)[0][0] 
    istab = np.where(Frbins_label == stab)[0][0] 
    binSC1 = binmap_SC1[isector,istab]
    binSC2 = binmap_SC2[isector,istab]
    binLT = binmap_MP5[isector,istab]
    WD_ref_north = WD_ref
    WD_ref_north[WD_ref>WDbins[-1]] = WD_ref_north[WD_ref>WDbins[-1]]-360
    labelSC1 = 'SC1 (' + str(len(binSC1)) + ')'
    labelSC2 = 'SC2 (' + str(len(binSC2)) + ')'
    labelLT = 'LT (' + str(len(binLT)) + ')'

    fig, axes = plt.subplots(2,3,figsize = (16,8))
    U_ref.reindex(binSC1).hist(ax = axes[0][0], bins = np.arange(Sbins[0],Sbins[1]+0.5,0.5), label = labelSC1, density=True, color = 'silver')
    U_ref.reindex(binLT).hist(ax = axes[0][0], bins = np.arange(Sbins[0],Sbins[1]+0.5,0.5), label = 'LT', fill=False, density=True, zorder=3)
    WD_ref_north.reindex(binSC1).hist(ax = axes[0][1], bins = np.arange(WDbins[isector],WDbins[isector+1]+0.5,0.5), label = labelSC1, density=True, color = 'silver')
    WD_ref_north.reindex(binLT).hist(ax = axes[0][1], bins = np.arange(WDbins[isector],WDbins[isector+1]+0.5,0.5), label = labelLT, fill=False, density=True, zorder=3)
    Fr_ref.reindex(binSC1).hist(ax = axes[0][2], bins = np.arange(Frbins[istab],Frbins[istab+1]+0.02,0.02), label = labelSC1, density=True, color = 'silver')
    Fr_ref.reindex(binLT).hist(ax = axes[0][2], bins = np.arange(Frbins[istab],Frbins[istab+1]+0.02,0.02), label = labelLT, fill=False, density=True, zorder=3)
    U_ref.reindex(binSC2).hist(ax = axes[1][0], bins = np.arange(Sbins[0],Sbins[1]+0.5,0.5), label = labelSC2, density=True, color = 'silver')
    U_ref.reindex(binLT).hist(ax = axes[1][0], bins = np.arange(Sbins[0],Sbins[1]+0.5,0.5), label = 'LT', fill=False, density=True, zorder=3)
    WD_ref_north.reindex(binSC2).hist(ax = axes[1][1], bins = np.arange(WDbins[isector],WDbins[isector+1]+0.5,0.5), label = labelSC2, density=True, color = 'silver')
    WD_ref_north.reindex(binLT).hist(ax = axes[1][1], bins = np.arange(WDbins[isector],WDbins[isector+1]+0.5,0.5), label = labelLT, fill=False, density=True, zorder=3)
    Fr_ref.reindex(binSC2).hist(ax = axes[1][2], bins = np.arange(Frbins[istab],Frbins[istab+1]+0.02,0.02), label = labelSC2, density=True, color = 'silver')
    Fr_ref.reindex(binLT).hist(ax = axes[1][2], bins = np.arange(Frbins[istab],Frbins[istab+1]+0.02,0.02), label = labelLT, fill=False, density=True, zorder=3)
    axes[0][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left'); axes[0][2].grid(zorder=0)
    axes[1][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left'); axes[1][2].grid(zorder=0)
    axes[1][0].set_xlabel('$U_{ref}$ [$m s^{-1}$]'); axes[1][0].grid(zorder=0)
    axes[1][1].set_xlabel('$WD_{ref}$ [$^{\circ}$]'); axes[1][1].grid(zorder=0)
    axes[1][2].set_xlabel('$Fr_{ref}^{-1}$'); axes[1][2].grid(zorder=0)
    axes[0][1].set_title('Sector: ' + sector + ', Stability: ' + stab); axes[0][1].grid(zorder=0)
    
def SC_mean_profiles(SC,sector,Frbins_label):
    """ 
    Plot mean profiles at all the mast of SC period
    """
    
    Nfr = len(Frbins_label)
    fig, axes = plt.subplots(2,3,figsize = (16,10))
    for mast in SC.keys():
        for fr in range(Nfr):
            stab = Frbins_label[fr]
            SC[mast].flow_correction_factor.loc[sector,stab].plot(y = 'z', ax = axes[0][fr], label = mast)
            SC[mast].turbulence_intensity.loc[sector,stab].plot(y = 'z', ax = axes[1][fr], label= mast)
    axes[0][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0][0].grid(); axes[0][0].set_xlim([0.6,1.1])
    axes[0][1].grid(); axes[0][1].set_xlim([0.6,1.1])
    axes[0][2].grid(); axes[0][2].set_xlim([0.6,1.1])
    axes[1][0].grid(); axes[1][0].set_xlim([0.03,0.17])
    axes[1][1].grid(); axes[1][1].set_xlim([0.03,0.17])
    axes[1][2].grid(); axes[1][2].set_xlim([0.03,0.17])
    
def SC_vs_LT_mean_profiles(SC1,SC2,LT,Frbins_label,sectors):
    """
    Compare short-term and long-term mean profiles at MP5
    """
    
    Nfr = len(Frbins_label)
    cmap = plt.get_cmap('bwr')
    Frcolors = np.flipud(cmap(np.linspace(0.,Nfr,Nfr)/Nfr))
    Frcolors[int((Nfr-1)/2),:] = np.array([0, 0, 0, 1])
    
    fig, axes = plt.subplots(2,3,figsize = (16,10))
    for fr in range(Nfr):
        stab = Frbins_label[fr]
        SC1['MP5'].wind_speed.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][0], 
                                                                       linestyle='-.', color = Frcolors[fr,:])
        SC2['MP5'].wind_speed.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][0], 
                                                                       linestyle=':', color = Frcolors[fr,:])
        LT['MP5'].wind_speed.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][0], 
                                                                      color = Frcolors[fr,:])
        SC1['MP5'].wind_speed.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][0], 
                                                                       linestyle='-.', color = Frcolors[fr,:])
        SC2['MP5'].wind_speed.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][0], 
                                                                       linestyle=':', color = Frcolors[fr,:])
        LT['MP5'].wind_speed.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][0], 
                                                                      color = Frcolors[fr,:])

        SC1['MP5'].flow_correction_factor.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][1], 
                                                                                   linestyle='-.', color = Frcolors[fr,:])
        SC2['MP5'].flow_correction_factor.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][1], 
                                                                                   linestyle=':', color = Frcolors[fr,:])
        LT['MP5'].flow_correction_factor.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][1], 
                                                                                  color = Frcolors[fr,:])
        SC1['MP5'].flow_correction_factor.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][1], 
                                                                                   linestyle='-.', color = Frcolors[fr,:])
        SC2['MP5'].flow_correction_factor.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][1], 
                                                                                   linestyle=':', color = Frcolors[fr,:])
        LT['MP5'].flow_correction_factor.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][1], 
                                                                                  color = Frcolors[fr,:])       
        
        SC1['MP5'].turbulence_intensity.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][2], 
                                            linestyle='-.', color = Frcolors[fr,:], label = 'SC1-' + Frbins_label[fr] + ' ('+ str(int(SC1['MP5'].samples.loc[sectors[0],stab].values.sum())) + ')')
        SC2['MP5'].turbulence_intensity.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][2], 
                                                                                 linestyle=':', color = Frcolors[fr,:], label = 'SC2-' + Frbins_label[fr] + ' ('+ str(int(SC2['MP5'].samples.loc[sectors[0],stab].values.sum())) + ')')
        LT['MP5'].turbulence_intensity.loc[sectors[0],stab].mean(axis = 0).plot(y = 'z', ax = axes[0][2], 
                                                                                color = Frcolors[fr,:], label = 'LT-' + Frbins_label[fr] + ' ('+ str(int(LT['MP5'].samples.loc[sectors[0],stab].values.sum())) + ')')
        SC1['MP5'].turbulence_intensity.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][2], 
                                                                                 linestyle='-.', color = Frcolors[fr,:], label = 'SC1-' + Frbins_label[fr] + ' ('+ str(int(SC1['MP5'].samples.loc[sectors[1],stab].values.sum())) + ')')
        SC2['MP5'].turbulence_intensity.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][2], 
                                                                                 linestyle=':', color = Frcolors[fr,:], label = 'SC2-' + Frbins_label[fr] + ' ('+ str(int(SC2['MP5'].samples.loc[sectors[1],stab].values.sum())) + ')')
        LT['MP5'].turbulence_intensity.loc[sectors[1],stab].mean(axis = 0).plot(y = 'z', ax = axes[1][2], 
                                                                                color = Frcolors[fr,:], label = 'LT-' + Frbins_label[fr] + ' ('+ str(int(LT['MP5'].samples.loc[sectors[1],stab].values.sum())) + ')')

    axes[0][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0][0].set_title('N sector: ' + ','.join([str(x) for x in sectors[0]])); axes[0][0].set_xlabel(''); axes[0][0].grid()
    axes[0][1].set_title('N sector: ' + ','.join([str(x) for x in sectors[0]])); axes[0][1].set_xlabel(''); axes[0][1].grid()
    axes[0][2].set_title('N sector: ' + ','.join([str(x) for x in sectors[0]])); axes[0][2].set_xlabel(''); axes[0][2].grid()
    axes[1][0].set_title('S sector: ' + ','.join([str(x) for x in sectors[1]])); axes[1][0].grid()
    axes[1][1].set_title('S sector: ' + ','.join([str(x) for x in sectors[1]])); axes[1][1].grid()
    axes[1][2].set_title('S sector: ' + ','.join([str(x) for x in sectors[1]])); axes[1][2].grid()
    
    axes[0][0].set_xlim([6, 12.5]); axes[1][0].set_xlim([6, 12.5])
    axes[0][1].set_xlim([0.84, 1.02]); axes[1][1].set_xlim([0.84, 1.02])
    axes[0][2].set_xlim([0.04, 0.13]); axes[1][2].set_xlim([0.04, 0.13]);

def xarray_global_attributes_from_yaml(xr, file_path):
    """
    Add global attributes to xarray xr based on yaml config file at file_path
    """
    try:
        with open(file_path, 'r') as stream:
            config = safe_load(stream)
            for key, value in config.items():
                xr.attrs[key] = value
    except FileNotFoundError as e:
        print('bad_config_path', file_path)
    except Exception as e:
        print('bad_config_formatting', str(e))
    return xr


def basemap_plot(src, masts, ref, ax, coord = 'utm'):
    # Add overviews to raster to plot faster at lower resolution (https://rasterio.readthedocs.io/en/latest/topics/overviews.html)
    #from rasterio.enums import Resampling
    #factors = [2, 4, 8, 16]
    #dst = rio.open('./inputs/DTM_Alaiz_2m.tif', 'r+')
    #dst.build_overviews(factors, Resampling.average)
    #dst.update_tags(ns='rio_overview', resampling='average')
    #dst.close()
    oview = src.overviews(1)[2] # choose overview (0 is largest, -1 is the smallest)
    topo = src.read(1, out_shape=(1, int(src.height // oview), int(src.width // oview)))
    if coord == 'xy':
        spatial_extent = [src.bounds.left - ref[0], src.bounds.right - ref[0], src.bounds.bottom - ref[1], src.bounds.top - ref[1]]
    else:
        spatial_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
    topo_ma = np.ma.masked_where(topo == 0 , topo, copy=True) 
#    ls = LightSource(azdeg=315, altdeg=60)
#    rgb = ls.shade(topo_ma, cmap=plt.cm.terrain, blend_mode='overlay') 
#    h_topo = ax.imshow(rgb, extent=spatial_extent, vmin=400, vmax=1200)
    h_topo = ax.imshow(topo_ma, cmap = plt.cm.terrain, extent=spatial_extent, vmin=300, vmax=1200)
    if coord == 'xy':
        h_masts = ax.scatter(masts['x'], masts['y'], s = 10, marker='s', c='k', label = 'Masts')
        for i, txt in enumerate(masts['Name']):
            ax.annotate(txt, (masts['x'][i]+50, masts['y'][i]+50))   
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    else:
        h_masts = ax.scatter(masts['easting[m]'], masts['northing[m]'], s = 10, marker='s', c='k', label = 'Masts')
        for i, txt in enumerate(masts['Name']):
            ax.annotate(txt, (masts['easting[m]'][i]+50, masts['northing[m]'][i]+50))   
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')      
            
    ax.set_title('ALEX17 sites')

    #ax.legend(handles = h_masts)
    plt.colorbar(h_topo, ax = ax)
    return h_masts

# Froude number
def froude(U,Lhill,zt1,zt2,Th1,Th2):
    """
    Computes the Froude number Fr based on a wind speed level U, the potential temperature gradient Th from two levels (zt1, zt2) and a length scale of the hill Lhill [m]
    We assume the same length scale for both stable and unstable conditions and follow the definition in Stull (1988) 
    """
    g = 9.8
    dThdz = np.abs(Th2-Th1)/(zt2-zt1) # potential temperature gradient
    Th_mean = 0.5*(Th2+Th1)
    Nbv = ((g/Th_mean)*dThdz)**0.5 # Brunt-Vaisala frequency [s-1]
    Fr = np.sign(Th2-Th1)*np.pi*U/(Nbv*Lhill) # Stull (1988)
    return Fr

# Potential temperature
def potential_temperature(T,p): 
    """
    Computes potential temperature from temperature T and pressure p [hPa] based on the Poisson equation
    """
    Th = np.multiply(T,(1000./p).pow(0.286))
    return Th

# 
def ensemble_averages(mast, binmap, masts_obs, WDbins_label, Frbins_label, levels):
    """
    Compute ensemble averages and standard deviations from a mast time series
    binmap: datetime indices to each bin  
    ds: returns xarray dataset  
    """
    Nwd = len(WDbins_label) # number of wind direction sectors
    Nfr = len(Frbins_label) # number of stability bins
    z = levels[mast]['zu']  # heights
    Nz = z.shape[0]         # number of vertical levels 

    S = np.empty((Nwd,Nfr,Nz))
    S_std = np.empty((Nwd,Nfr,Nz))
    I = np.empty((Nwd,Nfr,Nz))
    I_std = np.empty((Nwd,Nfr,Nz))
    FCF = np.empty((Nwd,Nfr,Nz))
    FCF_std = np.empty((Nwd,Nfr,Nz))
    N = np.empty((Nwd,Nfr))

    for wd in range(Nwd):
        for fr in range(Nfr):
            sector = WDbins_label[wd]
            stability = Frbins_label[fr]
            index = binmap[wd,fr]      
            S[wd,fr,:] = masts_obs[mast].reindex(index).iloc[:,levels[mast]['iu']].mean().values
            S_std[wd,fr,:] = masts_obs[mast].reindex(index).iloc[:,levels[mast]['iu']].std().values
            I[wd,fr,:] = masts_obs[mast].reindex(index).iloc[:,levels[mast]['iti']].mean().values
            I_std[wd,fr,:] = masts_obs[mast].reindex(index).iloc[:,levels[mast]['iti']].std().values
            FCF[wd,fr,:] = masts_obs[mast].reindex(index).iloc[:,levels[mast]['ifcf']].mean().values
            FCF_std[wd,fr,:] = masts_obs[mast].reindex(index).iloc[:,levels[mast]['ifcf']].std().values
            N[wd,fr] = len(index)

    ds = xr.Dataset(
        {
            "wind_speed": (("Sector", "Stability","z"), S),
            "wind_speed_std": (("Sector", "Stability","z"), S_std),
            "turbulence_intensity": (("Sector", "Stability","z"), I),
            "turbulence_intensity_std": (("Sector", "Stability","z"), I_std),
            "flow_correction_factor": (("Sector", "Stability","z"), FCF),
            "flow_correction_factor_std": (("Sector", "Stability","z"), FCF_std),
            "samples": (("Sector", "Stability"), N),
        },
        {"Sector": WDbins_label, "Stability": Frbins_label, "z": z},
        )
    ds.wind_speed.attrs['units'] = 'm s-1'
    ds.wind_speed_std.attrs['units'] = 'm s-1'
    ds.wind_speed.attrs['comments'] = 'mean value of binned wind speed'
    ds.wind_speed_std.attrs['comments'] = 'standard deviation of binned wind speed'
    ds.turbulence_intensity.attrs['comments'] = 'mean value of binned turbulence intensity'
    ds.turbulence_intensity_std.attrs['comments'] = 'standard deviation value of binned turbulence intensity'
    ds.flow_correction_factor.attrs['comments'] = 'mean value of binned flow correction factor'
    ds.flow_correction_factor_std.attrs['comments'] = 'standard deviation value of binned flow correction factor'
    ds.samples.attrs['comments'] = 'number of 10-min samples'
    
    return ds

def WD_vs_stab_bins(x,y,ts,statistic,bins,bins_label,plot = False):
    """
    Compute and plot distribution of samples per bin
    Inputs:
        - x: time-series of wind direction
        - y: time-series of stability
        - ts: time-series of values
        - statistic: which statistic to compute per bin ('count','mean','std')
        - bins: bin limits for x and y
        - bins_label: bin labels
        - plot: whether to plot the distribution or not
    Outputs: 
        - N_WDFr: dataframe wind bin sample count per wd and zL
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
    """
    
    Nwd, Nfr = [len(dim) for dim in bins_label]
    WDbins_label, Frbins_label = bins_label
    WDbins, Frbins = bins
    x = x.values.flatten()
    x[x>WDbins[-1]] = x[x>WDbins[-1]]-360
    y = y.values.flatten()
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, ts.values.flatten(), 
                                                                     statistic=statistic, 
                                                                     bins=bins, expand_binnumbers = True)
    N_WDFr = pd.DataFrame(statistic, index=WDbins_label, columns=Frbins_label)

    binmap = np.empty((Nwd, Nfr), dtype = object)
    for wd in range(Nwd):
        for fr in range(Nfr):
            binmap[wd,fr] = ts[np.logical_and(binnumber[0,:] == wd+1, binnumber[1,:] == fr+1)].index

    if plot:
        N_fr = np.sum(N_WDFr, axis = 0).rename('pdf')
        N_WD = np.sum(N_WDFr, axis = 1).rename('pdf')
        Nnorm_WDfr = N_WDFr.div(N_WD, axis=0)

        f1 = plt.figure(figsize = (18,8))
        cmap = plt.get_cmap('bwr')
        stabcolors = np.flipud(cmap(np.linspace(0.,Nfr,Nfr)/Nfr))
        ax1=Nnorm_WDfr.plot.bar(stacked=True, color=stabcolors, align='center', width=1.0, legend=False, 
                                rot=90, use_index = False, edgecolor='grey')
        ax2=(N_WD/N_WD.sum()).plot(ax=ax1, secondary_y=True, style='k',legend=False, rot=90, use_index = False);
        ax2.set_xticklabels(WDbins_label)
        #ax1.set_title('Wind direction vs stability')
        ax1.set_ylabel('$pdf_{norm}$($Fr^{-1}_{ref}$)')
        ax2.set_ylabel('$pdf$($WD_{ref}$)', rotation=-90, labelpad=15)
        ax1.set_yticks(np.linspace(0,1.,6))
        ax1.set_ylim([0,1.])
        ax2.set_yticks(np.linspace(0,0.2,6))

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.3, 1))
        
        #cellText = N_WDFr.T.astype(str).values.tolist() # add table to bar plot
        #the_table = plt.table(cellText=cellText,
        #                      rowLabels=zLbins_label,
        #                      rowColours=zLcolors,
        #                      colLabels=WDbins_label,
        #                      loc='bottom')

    return N_WDFr, binmap


def period_index(period, masts_obs, levels, period_name):
    """
    Returns datetime indices to for a period finding syncronized samples from quality-checked ("ok") levels of the masts operating within that period
    
    period: dictionary of masts
    masts_obs: dictionary of masts time-series dataframes
    levels: dictionary of masts levels (indices to time-series and heights)
    iperiod: indices to syncronized datetimes
    """

    iMP5 = ~masts_obs['MP5'].iloc[:,levels['MP5']['iok']].isnull().any(axis=1) # long-term period
    iMP5 = iMP5[iMP5].index
    iperiod = iMP5
    for mast in period.keys():
        imast = ~masts_obs[mast].iloc[:,levels[mast]['iok']].isnull().any(axis=1)
        imast = imast[imast].index
        iperiod = imast.intersection(iperiod)
        
    print(period_name +' period: from ' + str(iperiod[0]) + ' to ' + str(iperiod[-1]) + ', ' 
      + str(len(iperiod)) + ' 10-min samples (' +  
      "{:3.1f}".format(100*len(iperiod)/(6.*(iperiod[-1] - iperiod[0])/np.timedelta64(1, 'h'))) + '%)')
    return iperiod

def wind_shear(Ubottom,Utop,zbottom,ztop):
    """
    Power-law shear exponent  
    """
    alpha = np.log(Utop/Ubottom)/np.log(ztop/zbottom)
    return alpha

def get_levels():
    """
    Indices to measurement levels
    """
    levels = {
        'MP5': {'iu': np.array([0, 7, 14, 21, 28]), 
               'isu': np.array([1, 8, 15, 22, 29]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([32, 33, 34, 35, 36]),
               'it': np.array([37, 38, 39, 40, 41]),
               'ip': np.array([31]),
               'iti': np.array([42, 43, 44, 45, 46]),
               'ifcf': np.array([47, 48, 49, 50, 51]),
               'ialpha': np.array([52]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81, 38, 2])},
        'MP0': {'iu': np.array([0, 7, 14, 20, 27]), 
               'isu': np.array([1, 8, 15, 21, 28]), 
               'id': np.array([3, 10, 23]),
               'irh': np.array([]),
               'it': np.array([30, 31, 32]),
               'ip': np.array([33]),
               'iti': np.array([34, 35, 36, 37, 38]),
               'ifcf': np.array([39, 40, 41, 42, 43]),
               'ialpha': np.array([44]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 78]),
               'zt': np.array([113, 97, 81])},
        'MP1': {'iu': np.array([0, 7, 14, 21, 28]), 
               'isu': np.array([1, 8, 15, 22, 29]), 
               'id': np.array([3, 10, 24]),
               'irh': np.array([32, 33, 34]),
               'it': np.array([35, 36, 37]),
               'ip': np.array([31]),
               'iti': np.array([38, 39, 40, 41, 42]),
               'ifcf': np.array([43, 44, 45, 46, 47]),
               'ialpha': np.array([48]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 78]),
               'zt': np.array([113, 97, 81])},
        'MP3': {'iu': np.array([0, 7, 14, 21, 28]), 
               'isu': np.array([1, 8, 15, 22, 29]), 
               'id': np.array([3, 10, 24]),
               'irh': np.array([32, 33, 34]),
               'it': np.array([35, 36, 37]),
               'ip': np.array([31]),
               'iti': np.array([38, 39, 40, 41, 42]),
               'ifcf': np.array([43, 44, 45, 46, 47]),
               'ialpha': np.array([48]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 78]),
               'zt': np.array([113, 97, 81])},
        'MP6': {'iu': np.array([0, 7, 14, 21, 28]), 
               'isu': np.array([1, 8, 15, 22, 29]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([32, 33, 34]),
               'it': np.array([35, 36, 37]),
               'ip': np.array([31]),
               'iti': np.array([38, 39, 40, 41, 42]),
               'ifcf': np.array([43, 44, 45, 46, 47]),
               'ialpha': np.array([48]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81])},
        'A1': {'iu': np.array([0, 7, 14, 21]), 
               'isu': np.array([1, 8, 15, 22]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([28, 30, 32]),
               'it': np.array([29, 31, 33]),
               'ip': np.array([34]),
               'iti': np.array([35, 36, 37, 38]),
               'ifcf': np.array([39, 40, 41, 42]),
               'ialpha': np.array([43]),
               'zu': np.array([118, 102, 90, 78]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81])},
        'A2': {'iu': np.array([0, 7, 14, 21, 28]), 
               'isu': np.array([1, 8, 15, 22, 29]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([31, 33, 35]),
               'it': np.array([32, 34, 36]),
               'ip': np.array([37]),
               'iti': np.array([38, 39, 40, 41, 42]),
               'ifcf': np.array([43, 44, 45, 46, 47]),
               'ialpha': np.array([48]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 90, 78, 40]),
               'zt': np.array([113, 97, 81])},
        'A3': {'iu': np.array([0, 7, 14, 21]), 
               'isu': np.array([1, 8, 15, 22]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([]),
               'it': np.array([28, 29, 30]),
               'ip': np.array([31]),
               'iti': np.array([32, 33, 34, 35]),
               'ifcf': np.array([36, 37, 38, 39]),
               'ialpha': np.array([40]),
               'zu': np.array([118, 102, 90, 78]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81])},
        'A4': {'iu': np.array([0, 7, 14, 21, 28]), 
               'isu': np.array([1, 8, 15, 22, 29]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([]),
               'it': np.array([31, 32, 33]),
               'ip': np.array([34]),
               'iti': np.array([35, 36, 37, 38, 39]),
               'ifcf': np.array([40, 41, 42, 43, 44]),
               'ialpha': np.array([45]),
               'zu': np.array([118, 102, 90, 78, 40]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81])},
        'A5': {'iu': np.array([0, 7, 20, 27]), 
               'isu': np.array([1, 8, 14, 21]), 
               'id': np.array([3, 9, 16, 23]),
               'irh': np.array([]),
               'it': np.array([30, 31, 32]),
               'ip': np.array([33]),
               'iti': np.array([34, 35, 36, 37]),
               'ifcf': np.array([38, 39, 40, 41]),
               'ialpha': np.array([42]),
               'zu': np.array([118, 102, 90, 78]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81])},
        'A6': {'iu': np.array([0, 7, 14, 21]), 
               'isu': np.array([1, 8, 15, 22]), 
               'id': np.array([3, 10, 17, 24]),
               'irh': np.array([]),
               'it': np.array([28, 29, 30]),
               'ip': np.array([31]),
               'iti': np.array([32, 33, 34, 35]),
               'ifcf': np.array([36, 37, 38, 39]),
               'ialpha': np.array([40]),
               'zu': np.array([118, 102, 90, 78]),
               'zd': np.array([118, 102, 90, 78]),
               'zt': np.array([113, 97, 81])}}

    # These levels should all have good data to sync 
    levels['MP5']['iok'] = np.concatenate((levels['MP5']['iu'],levels['MP5']['isu'],
                                           levels['MP5']['id'][0].reshape((1,)),
                                           levels['MP5']['irh'][0].reshape((1,)),
                                           levels['MP5']['it'][1:3],
                                           levels['MP5']['ip'].reshape((1,))), axis=0)
    levels['MP6']['iok'] = np.concatenate((levels['MP6']['iu'],levels['MP6']['isu']), axis=0)
    levels['MP0']['iok'] = np.concatenate((levels['MP0']['iu'],levels['MP0']['isu']), axis=0)
    levels['MP1']['iok'] = np.concatenate((levels['MP1']['iu'],levels['MP1']['isu']), axis=0)
    levels['MP3']['iok'] = np.concatenate((levels['MP3']['iu'],levels['MP3']['isu']), axis=0)
    levels['A1']['iok'] = np.concatenate((levels['A1']['iu'],levels['A1']['isu']), axis=0)
    levels['A2']['iok'] = np.concatenate((levels['A2']['iu'],levels['A2']['isu']), axis=0)
    levels['A3']['iok'] = np.concatenate((levels['A3']['iu'],levels['A3']['isu']), axis=0)
    levels['A4']['iok'] = np.concatenate((levels['A4']['iu'],levels['A4']['isu']), axis=0)
    levels['A5']['iok'] = np.concatenate((levels['A5']['iu'],levels['A5']['isu']), axis=0)
    levels['A6']['iok'] = np.concatenate((levels['A6']['iu'],levels['A6']['isu']), axis=0)

    return levels

def read_csv_masts(files):
    """
    Read mast data from csv files
    """
    masts_obs = {}
    for file in files:
        filename = file + '.csv'
        mastname = file[0:3]
        df = pd.read_csv('./observations/' + filename)
        year = pd.DatetimeIndex(df['Fecha']).year
        month = pd.DatetimeIndex(df['Fecha']).month
        day = pd.DatetimeIndex(df['Fecha']).day
        hour = pd.DatetimeIndex(df['Hora']).hour
        minute = pd.DatetimeIndex(df['Hora']).minute
        second = pd.DatetimeIndex(df['Hora']).second
        date = np.column_stack((year,month,day,hour,minute,second))
        date = [datetime.datetime(*x) for x in date]
        df['time'] = date
        df = df.set_index('time')
        df = df.drop(['Fecha', 'Hora'], axis=1)
        masts_obs[mastname] = df 
        print(mastname + ': from ' + str(date[0]) + ' to ' + str(date[-1]))

    levels = get_levels()

    # Rename turbine positions
    masts_obs['A1'] = masts_obs.pop('MC1')
    masts_obs['A2'] = masts_obs.pop('MC2')
    masts_obs['A3'] = masts_obs.pop('MC3')
    masts_obs['A4'] = masts_obs.pop('MC4')
    masts_obs['A5'] = masts_obs.pop('MC5')
    masts_obs['A6'] = masts_obs.pop('MC6')

    masts_label = list(masts_obs.keys())

    # Remove duplicates (if any)
    for mast in masts_obs.keys():
        masts_obs[mast] = masts_obs[mast].loc[~masts_obs[mast].index.duplicated(keep='first')]

    # Replace nodata flag with NaNs
    nodata = -9999
    for mast in masts_obs.keys():
        masts_obs[mast] = masts_obs[mast].replace(nodata,np.nan)
        
    return masts_obs, levels

# From ACCESS mbd to csv files
#import sys, subprocess
#
#path = '/home/usuario/Documents/Desarrollo/Parque Experimental/Tratamiento de datos/'
#database = path + 'Alaiz_Nov12.mdb'
#n_mast = len(tbl_names)
#
## Read tables from access dababase file
#table_names = subprocess.Popen(['mdb-tables', '-1', database], stdout=subprocess.PIPE, text=True).communicate()[0]
#tables = table_names.split('\n')
#
## Walk through each table and dump as CSV file using 'mdb-export'
#for table in tables:
#    if table != '':
#        filename = table + '.csv'
#        print('Exporting ' + table)
#        with open(filename, 'wb') as f:
#            subprocess.check_call(['mdb-export', database, table], stdout=f)