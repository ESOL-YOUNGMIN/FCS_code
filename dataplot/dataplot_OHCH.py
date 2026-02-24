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


inputs = np.concatenate((norm_DataTrainSpec,norm_DataTestSpec, norm_DataValidSpec), axis=0)
targets1 = np.concatenate((FR_Train, FR_Test, FR_Valid), axis=0)
targets2 = np.concatenate((EQ_Train, EQ_Test, EQ_Valid), axis=0)
targets3 = np.concatenate((MIX_Train, MIX_Test, MIX_Valid), axis=0)

FR_label = ((targets1+1)*(max_FR-min_FR))/2+min_FR 
EQ_label = ((targets2+1)*(max_EQ-min_EQ))/2+min_EQ 
MIX_label = ((targets3+1)*(max_MIX-min_MIX))/2+min_MIX


target=np.zeros(len(targets1)*3).reshape(len(targets1),3)
target[:,0]=FR_label
target[:,1]=EQ_label
target[:,2]=MIX_label

indexindex=[]
for ii in range(0,9):    
    print(3.75*ii)
    for jj in range(0,7):
           lower_FR=105
           upper_FR=115
            
           lower_EQ=0.67+0.05*jj
           upper_EQ=0.72+0.05*jj
           lower_MIX=-1+3.75*ii
           upper_MIX=1+3.75*ii
        
           mask= (target[:,0]>=lower_FR) &(target[:,0]<upper_FR) &(target[:,1]>=lower_EQ) &(target[:,1]<upper_EQ) &(target[:,2]>=lower_MIX) &(target[:,2]<upper_MIX)
           find_index = np.where(mask)
           print(find_index[0][0])

index=[
63500,194000,67500,70500,152500,199500,201000
       ]
FR_OHCH=[]
for i in range(len(index)):
    OH=np.max(np.mean(inputs[index[i]:index[i]+300,200:300],axis=0))
    CH=np.max(np.mean(inputs[index[i]:index[i]+300,350:450],axis=0))
    OH_CH=OH/CH
    FR_OHCH.append(OH_CH)
    print('data:', index[i],'FR:',target[index[i],0],'EQ:',target[index[i],1],'MIX',target[index[i],2])
plt.plot(FR_OHCH)    



FR_OHCH=pd.DataFrame(FR_OHCH)
FR_OHCH.to_excel('./output/MIX_EQ_30.xlsx', index=False)


# .0
# 148500, 149000,194500,68000,71000,74000,75000

# 3.75
# 192500,149500,150500,68500,71500,198000,75500

# 7.5
# 61500,64000,195000,69000,72000,153000,76000

# 11.25
# 193000,64500,66500,151500,72500,74500,200000

# 15.0
# 193500,65000,67000,152000,197000,198500,76500

# 18.75
# 62000,150000,195500,69500,73000,153500,200500

# 22.5
# 62500,65500,151000,70000,73500,199000,77000

# 26.25
# 63000,66000,196000,196500,197500,154000,77500

# 30.0
# 63500,194000,67500,70500,152500,199500,201000

# 80
# 0,3000,177000,9000,13000,15500,17500
# 90
# 138000, 24000 ,27000, 30000 ,33000, 142000,37500
# 100
# 40000, 42500,44500,48500,188500,55500, 58500
# 110
# 148500,149000,194500,68000,71000,74000,75000
# 120
# 154500,80500,202000,157500,158500,159000,92500
# 130
# 96000,98000,100000,103500,105000,108000,110000
# 140
# 216000,168500,118500,171500,218500,219000,128000