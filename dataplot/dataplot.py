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


#dataplot
flowRate = [80, 90, 100, 110, 120, 130, 140]
equiRatio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
mixRatio = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5, 26.25, 30]

num=10
len_num=1
mean_spectra=np.zeros(1600*len_num).reshape(len_num,1600)
OHCH=np.zeros(9*7).reshape(9,7)
for MM in range(0,9):
    for EE in range(0,7):
        for index in range(0, len_num):
                print(index)
                filename0 = os.path.join(file_path_spectrum, str(flowRate[0]), str(equiRatio[EE]), str(mixRatio[MM]), "*.xlsx")
                d = os.listdir(os.path.dirname(filename0))
                filename = os.path.join(file_path_spectrum, str(flowRate[0]), str(equiRatio[EE]), str(mixRatio[MM]), d[0])
                spectra = pd.read_excel(filename, header=None).values
                spectra = spectra[5:1605, num+index]
                spectra[297] = (spectra[296] + spectra[298]) / 2.0
                spectra[1427] = (spectra[1426] + spectra[1428]) / 2.0
                mean_spectra[index,:]=spectra
           
        OH = np.max(np.mean(mean_spectra[:,200:210],axis=0))
        CH = np.max(np.mean(mean_spectra[:,350:450],axis=0))
        OHCH[MM,EE]=(OH/CH)
#plt.plot(OHCH)            
            
# fig, ax1 = plt.subplots()
# ax1.plot(FR_OHCH[0],label='80')
# ax1.plot(FR_OHCH[1],label='90')
# ax1.plot(FR_OHCH[2],label='100')
# ax1.plot(FR_OHCH[3],label='110')
# ax1.plot(FR_OHCH[4],label='120')
# ax1.plot(FR_OHCH[5],label='130')
# ax1.plot(FR_OHCH[6],label='140')
# ax1.legend()

fig, ax1 = plt.subplots()
ax1.plot(OHCH[0],label='0')
ax1.plot(OHCH[1],label='3.75')
ax1.plot(OHCH[2],label='7.5')
ax1.plot(OHCH[3],label='11.25')
ax1.plot(OHCH[4],label='15')
ax1.plot(OHCH[5],label='18.75')
ax1.plot(OHCH[6],label='22.5')
ax1.plot(OHCH[7],label='26.25')
ax1.plot(OHCH[8],label='30')
ax1.legend()


