#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Youngmin Cha
"""

# Paper: "Machine-learning-based prediction of key operational parameters in a
#         CH4/H2/air swirl combustor from a flame chemiluminescence spectrum"
# Terminology mapping:
#   FCS  = Flame Chemiluminescence Spectrum (coded as FES for legacy compatibility)
#   Vdot = Total combustion flow rate (V̇) [L/min] — coded as FR
#   phi  = Global equivalence ratio (φ) [-] — coded as EQ
#   XH2  = H2 blend ratio (X_H2) [mol%] — coded as MIX


from scipy import stats

flowRate = [80, 90, 100, 110, 120, 130, 140]
equiRatio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
mixRatio = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5, 26.25, 30]

FF = 6 
EE = 3 #0.85
#MM = 
EQ14085=[]
num_of_Files = 500
re_spectra = np.zeros(1600*500).reshape(500,1600)
for MM in range(0,8):
    filename0 = os.path.join(file_path_spectrum, str(flowRate[FF]), str(equiRatio[EE]), str(mixRatio[MM]), "*.xlsx")
    d = os.listdir(os.path.dirname(filename0))
    filename = os.path.join(file_path_spectrum, str(flowRate[FF]), str(equiRatio[EE]), str(mixRatio[MM]), d[0])
    spectra = pd.read_excel(filename, header=None).values
    spectra = spectra[5:1605, 1:num_of_Files+1]
    spectra[297] = (spectra[296] + spectra[298]) / 2.0
    spectra[1427] = (spectra[1426] + spectra[1428]) / 2.0
    
    
    df_spectra = pd.DataFrame(spectra)
    df_spectra = df_spectra.apply(pd.to_numeric, errors='coerce')
    
    spectra_zscores = np.abs(stats.zscore(df_spectra, axis=1))
    outliers = np.where(spectra_zscores > 5)
    
    spectra_mean_mask = df_spectra[(spectra_zscores <= 5).all(axis=1)]
    spectra_mean = np.mean(spectra_mean_mask, axis=1)
    
    outlier_spectra = outliers[1]
    outlier_features = outliers[0]
    
    print('outliner_num = ',len(outlier_spectra))
    print('outliner_num :' ,outlier_spectra)
    
    EQ14085.append(spectra_mean)
    # plt.plot(spectra[:,:])
    # plt.plot(spectra_mean,'r')
    # plt.show()
    plt.plot(spectra_mean,'b')
    plt.show()
    # for ii in range (0,len(outlier_spectra)):
    #     plt.plot(spectra[:,outlier_spectra[ii]])
    #     plt.plot(spectra_mean,'r')
    # plt.show()
    
for i in range(0,8):
    plt.plot(EQ14090[i],label=0.9)
    plt.plot(EQ14085[i],label=0.85)
    plt.plot(EQ14080[i],label=0.8)
    plt.xlabel('flow = 140')
    plt.legend()
    plt.show()

