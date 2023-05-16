./compiler.sh 

# -I [Sequential | Parallel] -B [Disable | Max] -S [Baseline] -M [None] 
# -T [Light | Heavy | Mix | ALL | LeNet | ResNet18 | VGG16 | GoogleNet] 

#./GPGPU -I Sequential -B Disable -S Baseline -M None -T Test1 
#./GPGPU -I Sequential -B Max     -S Baseline -M None -T Test1 
#./GPGPU -I Parallel   -B Disable -S Baseline -M None -T Test1 
#./GPGPU -I Parallel   -B Max     -S Baseline -M None -T Test1 

./GPGPU -I Sequential -B Disable -S Baseline -M None -T Test2 
./GPGPU -I Sequential -B Max     -S Baseline -M None -T Test2 
./GPGPU -I Parallel   -B Disable -S Baseline -M None -T Test2 
./GPGPU -I Parallel   -B Max     -S Baseline -M None -T Test2 
