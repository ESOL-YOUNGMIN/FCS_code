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



flowRate = [80, 90, 100, 110, 120, 130, 140]
equiRatio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
mixRatio = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5, 26.25, 30]
#num=50

#FR 80/ EQ 0.85/ MIX 0/
for MM in range(0,1):
    for FF in range(0,7):   
        #for index in range(0, num):
         #   print(index)
            filename0 = os.path.join(file_path_spectrum, str(flowRate[FF]), str(equiRatio[0]), str(mixRatio[MM]), "*.xlsx")
            d = os.listdir(os.path.dirname(filename0))
            filename = os.path.join(file_path_spectrum, str(flowRate[FF]), str(equiRatio[0]), str(mixRatio[MM]), d[0])
            spectra = pd.read_excel(filename, header=None).values
            spectra = spectra[5:1605, 1:501]
            spectra[297] = (spectra[296] + spectra[298]) / 2.0
            spectra[1427] = (spectra[1426] + spectra[1428]) / 2.0
            plt.plot(spectra)
            plt.xlabel(mixRatio[MM])

     #   plt.show()        
    
     