Short instruction to generate & fit Lasso

=============================
Sequence of  steps

basePath=/pscratch/sd/b/balewski/2026_causalNet_tmp2/

 mkdir plots/ spikesData/ truthDale/ lassoFdrFit

./gen_daleMatrices.py --basePath $basePath     --num_neurons 60 --num_excite 20     --spectral_radius 0.8  --idleRate 5 15 --num_steps 400_001 


Next step commands:
     basePath=/pscratch/sd/b/balewski/2026_causalNet_tmp2/
  ./view_daleMatrix.py  --basePath $basePath   --dataName daleN60_7fd00a  -p b -i 0   -X  -p a c d  
  ./view_spikesTrain.py  --basePath $basePath   --dataName daleN60_7fd00a  -p b -i 0   -X 
  ./gen_spikesTrain.py  --basePath $basePath   --truthName daleN60_7fd00a   


assuming data are generated
  ./fit_lassoPoisson.py  --basePath $basePath  --inpPath ${basePath}/truthDale --dataName
  daleN60_7fd00a   --num_epochs  200  


===========================
salloc -q interactive -C gpu  -t 4:00:00  -N 1 -A m2043
module load pytorch

basePath=/pscratch/sd/b/balewski/2026_causalNet_tmp2/
inpPath=${basePath}/truthDale

 ./fitLasso4GPU.sh   --basePath $basePath --inpPath $inpPath   --dataName daleN60_7fd00a  --num_epochs 100   --num_samples 100_001


1 GPU:
Epoch 20/200:  Loss_Tot=0.24109, only_L1=8.331e-05, Elapsed=17.7s

4 GPUS in parallel:
Epoch 20/100:  Loss_Tot=0.24156, only_L1=8.759e-05, Elapsed=9.4s

Mutiple bootstraps
basePath=/pscratch/sd/b/balewski/2026_causalNet_tmp2/
inpPath=${basePath}/truthDale

in-time:
./bigLassoBoots.sh  --basePath $basePath --inpPath $inpPath   --dataName daleN60_6cb864  --num_epochs 100    --num_samples 400_001  --dropDataFrac 0.33  --num_bootstraps 2   --bootsTag b2

de-sync
./bigLassoBoots.sh  --basePath $basePath --inpPath $inpPath   --dataName daleN60_6cb864  --num_epochs 100    --num_samples 400_001  --dropDataFrac 0.33  --num_bootstraps 2   --bootsTag b2   --desyncTime

