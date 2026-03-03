AA="/home/stu1/Dataset/TileSpGEMMDataset/cant.mtx"
base_name_=$(basename $AA .mtx)
ncu --set full \
    --target-processes all \
    --export /home/stu1/donghangcheng/AdaSpGEMM/profile/${base_name_}/cant_aat0_m16_n16_frac2.ncu-rep \
    ./bin_wtc_new/test_m16_n16_frac2 -d 0 -aat 0 $AA
# echo $AA
echo "${base_name_}_aat0_m16_n16_frac2 Finished!"

ncu --set full \
    --target-processes all \
    --export /home/stu1/donghangcheng/AdaSpGEMM/profile/${base_name_}/cant_aat0_m16_n16_frac5.ncu-rep \
    ./bin_wtc_new/test_m16_n16_frac5 -d 0 -aat 0 $AA
# echo $AA
echo "${base_name_}_aat0_m16_n16_frac5 Finished!"

ncu --set full \
    --target-processes all \
    --export /home/stu1/donghangcheng/AdaSpGEMM/profile/${base_name_}/cant_aat0_m16_n16_frac8.ncu-rep \
    ./bin_wtc_new/test_m16_n16_frac8 -d 0 -aat 0 $AA
# echo $AA
echo "${base_name_}_aat0_m16_n16_frac8 Finished!"

ncu --set full \
    --target-processes all \
    --export /home/stu1/donghangcheng/AdaSpGEMM/profile/${base_name_}/cant_aat0_m16_n16_frac11.ncu-rep \
    ./bin_wtc_new/test_m16_n16_frac11 -d 0 -aat 0 $AA
# echo $AA
echo "${base_name_}_aat0_m16_n16_frac11 Finished!"

ncu --set full \
    --target-processes all \
    --export /home/stu1/donghangcheng/AdaSpGEMM/profile/${base_name_}/cant_aat0_m16_n16_frac15.ncu-rep \
    ./bin_wtc_new/test_m16_n16_frac15 -d 0 -aat 0 $AA
# echo $AA
echo "${base_name_}_aat0_m16_n16_frac15 Finished!"