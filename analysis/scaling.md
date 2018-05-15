The models were transitioned from an earlier set with a potential temp of 1300 K.

For the model in the paper, I found the following adjustments to be in close agreement:

note the potential temp was hardcoded (1400 K)
#srun -n 48 python basic_model.py B 24 pd.adiabaticTempGrad*=0.9402271 pd.diffusionEnergyDepth*=1.061538 pd.diffusionVolumeDepth*=1.061538 md.res=192 md.faultThickness*=0.7 pd.viscosityFault*=0.45 
md.buoyancyFac*=1.1607 pd.yieldStressMax*=0.5 md.wedgeShallow*=1.0 md.depth*=1.25 md.aspectRatio=4.0

All the changes in the above CLAs are now hardcoded in the notebook. the md.buoyancyFac should now be unity. 