#!/bin/bash

# 사용 가능한 GPU 디바이스 설정 (,로 구분)
AVAILABLE_GPUS="0,1,2,3"  # 사용하고 싶은 GPU ID들을 여기에 설정

# GPU당 최대 동시 실행 프로세스 수 (메모리 부족 방지)
MAX_PROCESSES_PER_GPU=7

# GPU 목록을 배열로 변환
IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
MAX_CONCURRENT_JOBS=$((NUM_GPUS * MAX_PROCESSES_PER_GPU))

echo "Using ${NUM_GPUS} GPUs: ${AVAILABLE_GPUS}"
echo "Max ${MAX_PROCESSES_PER_GPU} processes per GPU (total: ${MAX_CONCURRENT_JOBS} concurrent jobs)"

# 실험 설정을 위한 associative array
declare -A experiments=(
    ["fair_cooking_05_2"]="cam03 lyra 0 48"  # 49 frames, 7494 7542
    ["fair_cooking_05_4"]="cam03 lyra 0 48"  # 49 frames, 5784 5832
    ["fair_cooking_05_6"]="cam03 lyra 0 48"  # 49 frames, 5036 5084
    ["fair_cooking_06_2"]="cam04 lyra 0 48"  # 49 frames, 6090 6138
    ["fair_cooking_06_4"]="cam04 lyra 0 48"  # 49 frames, 8469 8517
    ["fair_cooking_06_6"]="cam04 lyra 0 48"  # 49 frames, 5158 5206
    ["fair_cooking_07_2"]="cam01 lyra 0 48"  # 49 frames, 16905 16953
    ["fair_cooking_07_4"]="cam01 lyra 0 48"  # 49 frames, 6115 6163
    ["fair_cooking_08_10"]="cam01 lyra 0 48"  # 49 frames, 2956 3004
    ["fair_cooking_08_2"]="cam01 lyra 0 48"  # 49 frames, 4382 4430
    ["fair_cooking_08_4"]="cam01 lyra 0 48"  # 49 frames, 5079 5127
    ["fair_cooking_08_6"]="cam01 lyra 0 48"  # 49 frames, 3142 3190
    ["fair_cooking_08_8"]="cam01 lyra 0 48"  # 49 frames, 3412 3460
    ["georgiatech_cooking_01_01_2"]="cam01 lyra 0 48"  # 49 frames, 15373 15421
    ["georgiatech_cooking_01_01_4"]="cam01 lyra 0 48"  # 49 frames, 16343 16391
    ["georgiatech_cooking_01_02_2"]="cam01 lyra 0 48"  # 49 frames, 20158 20206
    ["georgiatech_cooking_01_02_4"]="cam01 lyra 0 48"  # 49 frames, 13594 13642
    ["georgiatech_cooking_01_03_2"]="cam02 lyra 0 48"  # 49 frames, 18685 18733
    ["georgiatech_cooking_01_03_4"]="cam02 lyra 0 48"  # 49 frames, 11257 11305
    ["georgiatech_cooking_04_01_2"]="cam02 lyra 0 48"  # 49 frames, 31045 31093
    ["georgiatech_cooking_04_02_2"]="cam01 lyra 0 48"  # 49 frames, 22851 22899
    ["georgiatech_cooking_04_02_4"]="cam03 lyra 0 48"  # 49 frames, 17485 17533
    ["georgiatech_cooking_07_03_2"]="cam01 lyra 0 48"  # 49 frames, 20446 20494
    ["georgiatech_cooking_09_01_3"]="cam03 lyra 0 48"  # 49 frames, 17307 17355
    ["georgiatech_cooking_09_01_6"]="cam05 lyra 0 48"  # 49 frames, 13540 13588
    ["georgiatech_cooking_09_02_2"]="cam05 lyra 0 48"  # 49 frames, 18546 18594
    ["georgiatech_cooking_09_02_4"]="cam01 lyra 0 48"  # 49 frames, 13693 13741
    ["georgiatech_cooking_10_01_2"]="cam01 lyra 0 48"  # 49 frames, 23121 23169
    ["georgiatech_cooking_10_01_6"]="cam01 lyra 0 48"  # 49 frames, 14128 14176
    ["georgiatech_cooking_10_02_2"]="cam05 lyra 0 48"  # 49 frames, 20907 20955
    ["georgiatech_cooking_10_02_4"]="cam01 lyra 0 48"  # 49 frames, 16333 16381
    ["georgiatech_cooking_11_01_4"]="cam05 lyra 0 48"  # 49 frames, 20599 20647
    ["georgiatech_cooking_11_01_6"]="cam05 lyra 0 48"  # 49 frames, 14281 14329
    ["georgiatech_cooking_11_02_3"]="cam01 lyra 0 48"  # 49 frames, 18082 18130
    ["georgiatech_cooking_14_01_2"]="cam01 lyra 0 48"  # 49 frames, 14802 14850
    ["georgiatech_cooking_14_01_6"]="cam01 lyra 0 48"  # 49 frames, 11438 11486
    ["georgiatech_cooking_14_02_2"]="cam04 lyra 0 48"  # 49 frames, 14688 14736
    ["georgiatech_cooking_14_02_7"]="cam01 lyra 0 48"  # 49 frames, 12290 12338
    ["georgiatech_cooking_14_03_2"]="cam01 lyra 0 48"  # 49 frames, 22465 22513
    ["iiith_cooking_01_1"]="cam01 lyra 0 48"  # 49 frames, 13029 13077
    ["iiith_cooking_02_1"]="cam01 lyra 0 48"  # 49 frames, 7494 7542
    ["iiith_cooking_02_3"]="cam01 lyra 0 48"  # 49 frames, 5211 5259
    ["iiith_cooking_03_1"]="cam02 lyra 0 48"  # 49 frames, 4447 4495
    ["iiith_cooking_03_3"]="cam01 lyra 0 48"  # 49 frames, 4347 4395
    ["iiith_cooking_04_1"]="cam02 lyra 0 48"  # 49 frames, 6007 6055
    ["iiith_cooking_05_1"]="cam02 lyra 0 48"  # 49 frames, 3397 3445
    ["iiith_cooking_05_3"]="cam02 lyra 0 48"  # 49 frames, 2868 2916
    ["iiith_cooking_05_5"]="cam02 lyra 0 48"  # 49 frames, 4629 4677
    ["iiith_cooking_06_1"]="cam01 lyra 0 48"  # 49 frames, 4237 4285
    ["iiith_cooking_07_1"]="cam02 lyra 0 48"  # 49 frames, 2342 2390
    ["iiith_cooking_08_1"]="cam02 lyra 0 48"  # 49 frames, 2405 2453
    ["iiith_cooking_100_2"]="cam02 lyra 0 48"  # 49 frames, 3208 3256
    ["iiith_cooking_100_4"]="cam02 lyra 0 48"  # 49 frames, 982 1030
    ["iiith_cooking_108_2"]="cam01 lyra 0 48"  # 49 frames, 1884 1932
    ["iiith_cooking_108_4"]="cam04 lyra 0 48"  # 49 frames, 727 775
    ["iiith_cooking_108_5"]="cam01 lyra 0 48"  # 49 frames, 814 862
    ["iiith_cooking_109_2"]="cam01 lyra 0 48"  # 49 frames, 1771 1819
    ["iiith_cooking_109_4"]="cam04 lyra 0 48"  # 49 frames, 409 457
    ["iiith_cooking_109_5"]="cam02 lyra 0 48"  # 49 frames, 622 670
    ["iiith_cooking_10_1"]="cam01 lyra 0 48"  # 49 frames, 2579 2627
    ["iiith_cooking_110_2"]="cam01 lyra 0 48"  # 49 frames, 1652 1700
    ["iiith_cooking_110_4"]="cam01 lyra 0 48"  # 49 frames, 996 1044
    ["iiith_cooking_111_2"]="cam01 lyra 0 48"  # 49 frames, 1497 1545
    ["iiith_cooking_111_4"]="cam04 lyra 0 48"  # 49 frames, 633 681
    ["iiith_cooking_111_5"]="cam01 lyra 0 48"  # 49 frames, 710 758
    ["iiith_cooking_112_2"]="cam01 lyra 0 48"  # 49 frames, 2071 2119
    ["iiith_cooking_112_3"]="cam04 lyra 0 48"  # 49 frames, 580 628
    ["iiith_cooking_112_4"]="cam01 lyra 0 48"  # 49 frames, 1022 1070
    ["iiith_cooking_115_2"]="cam01 lyra 0 48"  # 49 frames, 4191 4239
    ["iiith_cooking_115_3"]="cam02 lyra 0 48"  # 49 frames, 1355 1403
    ["iiith_cooking_115_5"]="cam01 lyra 0 48"  # 49 frames, 1957 2005
    ["iiith_cooking_116_2"]="cam01 lyra 0 48"  # 49 frames, 4065 4113
    ["iiith_cooking_116_4"]="cam01 lyra 0 48"  # 49 frames, 1419 1467
    ["iiith_cooking_116_5"]="cam01 lyra 0 48"  # 49 frames, 1828 1876
    ["iiith_cooking_117_2"]="cam01 lyra 0 48"  # 49 frames, 2976 3024
    ["iiith_cooking_117_4"]="cam01 lyra 0 48"  # 49 frames, 2309 2357
    ["iiith_cooking_118_2"]="cam01 lyra 0 48"  # 49 frames, 2017 2065
    ["iiith_cooking_11_1"]="cam01 lyra 0 48"  # 49 frames, 5046 5094
    ["iiith_cooking_122_2"]="cam01 lyra 0 48"  # 49 frames, 2508 2556
    ["iiith_cooking_122_4"]="cam01 lyra 0 48"  # 49 frames, 1345 1393
    ["iiith_cooking_123_2"]="cam03 lyra 0 48"  # 49 frames, 5428 5476
    ["iiith_cooking_123_4"]="cam03 lyra 0 48"  # 49 frames, 3275 3323
    ["iiith_cooking_123_6"]="cam02 lyra 0 48"  # 49 frames, 2224 2272
    ["iiith_cooking_124_2"]="cam03 lyra 0 48"  # 49 frames, 2964 3012
    ["iiith_cooking_124_4"]="cam01 lyra 0 48"  # 49 frames, 1828 1876
    ["iiith_cooking_124_5"]="cam01 lyra 0 48"  # 49 frames, 1431 1479
    ["iiith_cooking_128_2"]="cam03 lyra 0 48"  # 49 frames, 3187 3235
    ["iiith_cooking_128_4"]="cam03 lyra 0 48"  # 49 frames, 2822 2870
    ["iiith_cooking_12_1"]="cam02 lyra 0 48"  # 49 frames, 4078 4126
    ["iiith_cooking_130_2"]="cam01 lyra 0 48"  # 49 frames, 5038 5086
    ["iiith_cooking_130_4"]="cam03 lyra 0 48"  # 49 frames, 5060 5108
    ["iiith_cooking_131_2"]="cam01 lyra 0 48"  # 49 frames, 5980 6028
    ["iiith_cooking_131_4"]="cam03 lyra 0 48"  # 49 frames, 5507 5555
    ["iiith_cooking_132_4"]="cam01 lyra 0 48"  # 49 frames, 4568 4616
    ["iiith_cooking_134_2"]="cam01 lyra 0 48"  # 49 frames, 6877 6925
    ["iiith_cooking_135_2"]="cam01 lyra 0 48"  # 49 frames, 5833 5881
    ["iiith_cooking_136_2"]="cam01 lyra 0 48"  # 49 frames, 6268 6316
    ["iiith_cooking_13_1"]="cam01 lyra 0 48"  # 49 frames, 6850 6898
    ["iiith_cooking_141_2"]="cam02 lyra 0 48"  # 49 frames, 5184 5232
    ["iiith_cooking_142_2"]="cam01 lyra 0 48"  # 49 frames, 3544 3592
    ["iiith_cooking_143_2"]="cam01 lyra 0 48"  # 49 frames, 4254 4302
    ["iiith_cooking_146_2"]="cam01 lyra 0 48"  # 49 frames, 4124 4172
    ["iiith_cooking_147_2"]="cam01 lyra 0 48"  # 49 frames, 4994 5042
    ["iiith_cooking_148_2"]="cam01 lyra 0 48"  # 49 frames, 6770 6818
    ["iiith_cooking_149_2"]="cam02 lyra 0 48"  # 49 frames, 6847 6895
    ["iiith_cooking_14_1"]="cam02 lyra 0 48"  # 49 frames, 5610 5658
    ["iiith_cooking_15_1"]="cam01 lyra 0 48"  # 49 frames, 4081 4129
    ["iiith_cooking_16_1"]="cam01 lyra 0 48"  # 49 frames, 5872 5920
    ["iiith_cooking_17_1"]="cam01 lyra 0 48"  # 49 frames, 4582 4630
    ["iiith_cooking_18_1"]="cam01 lyra 0 48"  # 49 frames, 3780 3828
    ["iiith_cooking_19_1"]="cam01 lyra 0 48"  # 49 frames, 4952 5000
    ["iiith_cooking_20_1"]="cam01 lyra 0 48"  # 49 frames, 3106 3154
    ["iiith_cooking_22_1"]="cam01 lyra 0 48"  # 49 frames, 3446 3494
    ["iiith_cooking_23_1"]="cam03 lyra 0 48"  # 49 frames, 2501 2549
    ["iiith_cooking_24_1"]="cam01 lyra 0 48"  # 49 frames, 2302 2350
    ["iiith_cooking_25_1"]="cam01 lyra 0 48"  # 49 frames, 3458 3506
    ["iiith_cooking_25_3"]="cam03 lyra 0 48"  # 49 frames, 1527 1575
    ["iiith_cooking_26_1"]="cam01 lyra 0 48"  # 49 frames, 4207 4255
    ["iiith_cooking_27_1"]="cam01 lyra 0 48"  # 49 frames, 3654 3702
    ["iiith_cooking_27_3"]="cam03 lyra 0 48"  # 49 frames, 1519 1567
    ["iiith_cooking_28_1"]="cam03 lyra 0 48"  # 49 frames, 4915 4963
    ["iiith_cooking_40_1"]="cam01 lyra 0 48"  # 49 frames, 3028 3076
    ["iiith_cooking_41_1"]="cam01 lyra 0 48"  # 49 frames, 3010 3058
    ["iiith_cooking_41_3"]="cam02 lyra 0 48"  # 49 frames, 1422 1470
    ["iiith_cooking_42_1"]="cam01 lyra 0 48"  # 49 frames, 2594 2642
    ["iiith_cooking_43_1"]="cam02 lyra 0 48"  # 49 frames, 2802 2850
    ["iiith_cooking_43_3"]="cam02 lyra 0 48"  # 49 frames, 1660 1708
    ["iiith_cooking_44_1"]="cam01 lyra 0 48"  # 49 frames, 2391 2439
    ["iiith_cooking_45_1"]="cam02 lyra 0 48"  # 49 frames, 2857 2905
    ["iiith_cooking_45_3"]="cam04 lyra 0 48"  # 49 frames, 1816 1864
    ["iiith_cooking_56_2"]="cam01 lyra 0 48"  # 49 frames, 4065 4113
    ["iiith_cooking_57_2"]="cam01 lyra 0 48"  # 49 frames, 2683 2731
    ["iiith_cooking_57_4"]="cam01 lyra 0 48"  # 49 frames, 1771 1819
    ["iiith_cooking_71_2"]="cam01 lyra 0 48"  # 49 frames, 4211 4259
    ["iiith_cooking_71_4"]="cam01 lyra 0 48"  # 49 frames, 1422 1470
    ["iiith_cooking_71_6"]="cam02 lyra 0 48"  # 49 frames, 2196 2244
    ["iiith_cooking_72_2"]="cam01 lyra 0 48"  # 49 frames, 3669 3717
    ["iiith_cooking_72_4"]="cam01 lyra 0 48"  # 49 frames, 1088 1136
    ["iiith_cooking_72_6"]="cam02 lyra 0 48"  # 49 frames, 1346 1394
    ["iiith_cooking_74_2"]="cam02 lyra 0 48"  # 49 frames, 3518 3566
    ["iiith_cooking_74_4"]="cam01 lyra 0 48"  # 49 frames, 1302 1350
    ["iiith_cooking_74_6"]="cam02 lyra 0 48"  # 49 frames, 1394 1442
    ["iiith_cooking_75_2"]="cam01 lyra 0 48"  # 49 frames, 4029 4077
    ["iiith_cooking_76_2"]="cam01 lyra 0 48"  # 49 frames, 1174 1222
    ["iiith_cooking_76_4"]="cam02 lyra 0 48"  # 49 frames, 1318 1366
    ["iiith_cooking_77_2"]="cam02 lyra 0 48"  # 49 frames, 2266 2314
    ["iiith_cooking_78_2"]="cam01 lyra 0 48"  # 49 frames, 1155 1203
    ["iiith_cooking_79_2"]="cam01 lyra 0 48"  # 49 frames, 2296 2344
    ["iiith_cooking_80_2"]="cam01 lyra 0 48"  # 49 frames, 886 934
    ["iiith_cooking_81_2"]="cam01 lyra 0 48"  # 49 frames, 2595 2643
    ["iiith_cooking_82_2"]="cam01 lyra 0 48"  # 49 frames, 583 631
    ["iiith_cooking_84_2"]="cam01 lyra 0 48"  # 49 frames, 859 907
    ["iiith_cooking_84_4"]="cam01 lyra 0 48"  # 49 frames, 1069 1117
    ["iiith_cooking_85_2"]="cam02 lyra 0 48"  # 49 frames, 6325 6373
    ["iiith_cooking_87_2"]="cam02 lyra 0 48"  # 49 frames, 4030 4078
    ["iiith_cooking_87_4"]="cam02 lyra 0 48"  # 49 frames, 1747 1795
    ["iiith_cooking_87_6"]="cam01 lyra 0 48"  # 49 frames, 1540 1588
    ["iiith_cooking_91_2"]="cam02 lyra 0 48"  # 49 frames, 3192 3240
    ["iiith_cooking_92_2"]="cam02 lyra 0 48"  # 49 frames, 1390 1438
    ["iiith_cooking_93_2"]="cam01 lyra 0 48"  # 49 frames, 3000 3048
    ["iiith_cooking_93_4"]="cam02 lyra 0 48"  # 49 frames, 1209 1257
    ["iiith_cooking_93_6"]="cam01 lyra 0 48"  # 49 frames, 1234 1282
    ["iiith_cooking_96_2"]="cam04 lyra 0 48"  # 49 frames, 3443 3491
    ["iiith_cooking_97_2"]="cam04 lyra 0 48"  # 49 frames, 2095 2143
    ["iiith_cooking_98_2"]="cam02 lyra 0 48"  # 49 frames, 3153 3201
    ["iiith_cooking_98_5"]="cam04 lyra 0 48"  # 49 frames, 1420 1468
    ["iiith_cooking_98_7"]="cam02 lyra 0 48"  # 49 frames, 1854 1902
    ["iiith_cooking_99_2"]="cam01 lyra 0 48"  # 49 frames, 2686 2734
    ["iiith_cooking_99_4"]="cam04 lyra 0 48"  # 49 frames, 1150 1198
    ["indiana_cooking_01_3"]="cam02 lyra 0 48"  # 49 frames, 6604 6652
    ["indiana_cooking_03_2"]="cam02 lyra 0 48"  # 49 frames, 5344 5392
    ["indiana_cooking_03_3"]="cam02 lyra 0 48"  # 49 frames, 7539 7587
    ["indiana_cooking_04_2"]="cam02 lyra 0 48"  # 49 frames, 7571 7619
    ["indiana_cooking_04_3"]="cam02 lyra 0 48"  # 49 frames, 9198 9246
    ["indiana_cooking_10_2"]="cam03 lyra 0 48"  # 49 frames, 6477 6525
    ["indiana_cooking_10_5"]="cam03 lyra 0 48"  # 49 frames, 14181 14229
    ["indiana_cooking_12_2"]="cam03 lyra 0 48"  # 49 frames, 7353 7401
    ["indiana_cooking_12_3"]="cam04 lyra 0 48"  # 49 frames, 15723 15771
    ["indiana_cooking_12_4"]="cam03 lyra 0 48"  # 49 frames, 20163 20211
    ["indiana_cooking_13_2"]="cam03 lyra 0 48"  # 49 frames, 14023 14071
    ["indiana_cooking_14_3"]="cam04 lyra 0 48"  # 49 frames, 8483 8531
    ["indiana_cooking_14_4"]="cam03 lyra 0 48"  # 49 frames, 14588 14636
    ["indiana_cooking_14_5"]="cam04 lyra 0 48"  # 49 frames, 16009 16057
    ["indiana_cooking_20_2"]="cam03 lyra 0 48"  # 49 frames, 6868 6916
    ["indiana_cooking_20_3"]="cam03 lyra 0 48"  # 49 frames, 11197 11245
    ["indiana_cooking_20_4"]="cam03 lyra 0 48"  # 49 frames, 16414 16462
    ["indiana_cooking_20_5"]="cam04 lyra 0 48"  # 49 frames, 14209 14257
    ["indiana_cooking_21_2"]="cam03 lyra 0 48"  # 49 frames, 13683 13731
    ["indiana_cooking_21_3"]="cam03 lyra 0 48"  # 49 frames, 13922 13970
    ["indiana_cooking_21_4"]="cam03 lyra 0 48"  # 49 frames, 12505 12553
    ["indiana_cooking_21_5"]="cam03 lyra 0 48"  # 49 frames, 13146 13194
    ["indiana_cooking_22_2"]="cam03 lyra 0 48"  # 49 frames, 7807 7855
    ["indiana_cooking_22_3"]="cam03 lyra 0 48"  # 49 frames, 13276 13324
    ["indiana_cooking_22_4"]="cam03 lyra 0 48"  # 49 frames, 12227 12275
    ["indiana_cooking_22_5"]="cam03 lyra 0 48"  # 49 frames, 12430 12478
    ["indiana_cooking_26_2"]="cam03 lyra 0 48"  # 49 frames, 8629 8677
    ["indiana_cooking_26_3"]="cam03 lyra 0 48"  # 49 frames, 15922 15970
    ["indiana_cooking_26_4"]="cam03 lyra 0 48"  # 49 frames, 21877 21925
    ["indiana_cooking_27_2"]="cam04 lyra 0 48"  # 49 frames, 20340 20388
    ["minnesota_cooking_021_2"]="cam02 lyra 0 48"  # 49 frames, 14715 14763
    ["minnesota_cooking_022_2"]="cam02 lyra 0 48"  # 49 frames, 16838 16886
    ["minnesota_cooking_022_4"]="cam03 lyra 0 48"  # 49 frames, 12949 12997
    ["minnesota_cooking_023_2"]="cam04 lyra 0 48"  # 49 frames, 6610 6658
    ["minnesota_cooking_023_4"]="cam04 lyra 0 48"  # 49 frames, 4245 4293
    ["minnesota_cooking_024_2"]="cam04 lyra 0 48"  # 49 frames, 8346 8394
    ["minnesota_cooking_030_2"]="cam03 lyra 0 48"  # 49 frames, 7448 7496
    ["minnesota_cooking_030_5"]="cam03 lyra 0 48"  # 49 frames, 19195 19243
    ["minnesota_cooking_031_4"]="cam03 lyra 0 48"  # 49 frames, 19082 19130
    ["minnesota_cooking_032_2"]="cam03 lyra 0 48"  # 49 frames, 13696 13744
    ["minnesota_cooking_032_4"]="cam01 lyra 0 48"  # 49 frames, 7008 7056
    ["minnesota_cooking_050_2"]="cam01 lyra 0 48"  # 49 frames, 8466 8514
    ["minnesota_cooking_050_4"]="cam04 lyra 0 48"  # 49 frames, 21711 21759
    ["minnesota_cooking_074_2"]="cam04 lyra 0 48"  # 49 frames, 15077 15125
    ["nus_cooking_06_2"]="cam03 lyra 0 48"  # 49 frames, 4028 4076
    ["nus_cooking_06_3"]="cam03 lyra 0 48"  # 49 frames, 6541 6589
    ["nus_cooking_06_4"]="cam02 lyra 0 48"  # 49 frames, 5786 5834
    ["nus_cooking_06_5"]="cam02 lyra 0 48"  # 49 frames, 1141 1189
    ["nus_cooking_07_2"]="cam02 lyra 0 48"  # 49 frames, 2734 2782
    ["nus_cooking_07_3"]="cam03 lyra 0 48"  # 49 frames, 6483 6531
    ["nus_cooking_07_4"]="cam02 lyra 0 48"  # 49 frames, 5364 5412
    ["nus_cooking_07_5"]="cam02 lyra 0 48"  # 49 frames, 1240 1288
    ["nus_cooking_10_2"]="cam03 lyra 0 48"  # 49 frames, 3985 4033
    ["nus_cooking_10_3"]="cam03 lyra 0 48"  # 49 frames, 6091 6139
    ["nus_cooking_10_4"]="cam02 lyra 0 48"  # 49 frames, 6586 6634
    ["nus_cooking_10_5"]="cam03 lyra 0 48"  # 49 frames, 1198 1246
    ["nus_cooking_16_2"]="cam02 lyra 0 48"  # 49 frames, 3670 3718
    ["nus_cooking_16_3"]="cam02 lyra 0 48"  # 49 frames, 9665 9713
    ["nus_cooking_16_4"]="cam02 lyra 0 48"  # 49 frames, 7249 7297
    ["nus_cooking_16_5"]="cam03 lyra 0 48"  # 49 frames, 2001 2049
    ["nus_cooking_17_2"]="cam02 lyra 0 48"  # 49 frames, 2959 3007
    ["nus_cooking_17_3"]="cam02 lyra 0 48"  # 49 frames, 4581 4629
    ["nus_cooking_17_4"]="cam02 lyra 0 48"  # 49 frames, 4846 4894
    ["nus_cooking_17_5"]="cam02 lyra 0 48"  # 49 frames, 971 1019
    ["sfu_cooking015_2"]="cam04 lyra 0 48"  # 49 frames, 2425 2473
    ["sfu_cooking015_4"]="cam02 lyra 0 48"  # 49 frames, 2542 2590
    ["sfu_cooking017_2"]="cam04 lyra 0 48"  # 49 frames, 1162 1210
    ["sfu_cooking017_4"]="cam04 lyra 0 48"  # 49 frames, 959 1007
    ["sfu_cooking017_6"]="cam03 lyra 0 48"  # 49 frames, 4119 4167
    ["sfu_cooking017_9"]="cam03 lyra 0 48"  # 49 frames, 2569 2617
    ["sfu_cooking020_2"]="cam01 lyra 0 48"  # 49 frames, 1780 1828
    ["sfu_cooking020_4"]="cam01 lyra 0 48"  # 49 frames, 1537 1585
    ["sfu_cooking020_6"]="cam01 lyra 0 48"  # 49 frames, 4102 4150
    ["sfu_cooking020_8"]="cam01 lyra 0 48"  # 49 frames, 3295 3343
    ["sfu_cooking023_2"]="cam04 lyra 0 48"  # 49 frames, 795 843
    ["sfu_cooking023_4"]="cam03 lyra 0 48"  # 49 frames, 570 618
    ["sfu_cooking023_6"]="cam04 lyra 0 48"  # 49 frames, 6205 6253
    ["sfu_cooking023_8"]="cam03 lyra 0 48"  # 49 frames, 3649 3697
    ["sfu_cooking025_2"]="cam04 lyra 0 48"  # 49 frames, 408 456
    ["sfu_cooking025_3"]="cam01 lyra 0 48"  # 49 frames, 592 640
    ["sfu_cooking025_5"]="cam04 lyra 0 48"  # 49 frames, 1756 1804
    ["sfu_cooking025_7"]="cam02 lyra 0 48"  # 49 frames, 223 271
    ["sfu_cooking026_2"]="cam04 lyra 0 48"  # 49 frames, 2892 2940
    ["sfu_cooking026_4"]="cam04 lyra 0 48"  # 49 frames, 788 836
    ["sfu_cooking026_6"]="cam04 lyra 0 48"  # 49 frames, 572 620
    ["sfu_cooking026_8"]="cam04 lyra 0 48"  # 49 frames, 286 334
    ["sfu_cooking029_2"]="cam02 lyra 0 48"  # 49 frames, 4780 4828
    ["sfu_cooking029_4"]="cam04 lyra 0 48"  # 49 frames, 5008 5056
    ["sfu_cooking029_6"]="cam04 lyra 0 48"  # 49 frames, 9760 9808
    ["sfu_cooking029_8"]="cam02 lyra 0 48"  # 49 frames, 4011 4059
    ["sfu_cooking031_1"]="cam03 lyra 0 48"  # 49 frames, 4828 4876
    ["sfu_cooking031_10"]="cam01 lyra 0 48"  # 49 frames, 496 544
    ["sfu_cooking031_3"]="cam01 lyra 0 48"  # 49 frames, 3458 3506
    ["sfu_cooking031_5"]="cam03 lyra 0 48"  # 49 frames, 15365 15413
    ["sfu_cooking031_7"]="cam03 lyra 0 48"  # 49 frames, 7376 7424
    ["sfu_cooking031_9"]="cam03 lyra 0 48"  # 49 frames, 5704 5752
    ["sfu_cooking032_1"]="cam03 lyra 0 48"  # 49 frames, 5413 5461
    ["sfu_cooking032_3"]="cam01 lyra 0 48"  # 49 frames, 3531 3579
    ["sfu_cooking032_4"]="cam01 lyra 0 48"  # 49 frames, 4000 4048
    ["sfu_cooking_001_2"]="cam02 lyra 0 48"  # 49 frames, 3724 3772
    ["sfu_cooking_001_4"]="cam02 lyra 0 48"  # 49 frames, 2428 2476
    ["sfu_cooking_001_6"]="cam04 lyra 0 48"  # 49 frames, 3535 3583
    ["sfu_cooking_001_8"]="cam02 lyra 0 48"  # 49 frames, 3687 3735
    ["sfu_cooking_002_1"]="cam05 lyra 0 48"  # 49 frames, 5894 5942
    ["sfu_cooking_002_3"]="cam05 lyra 0 48"  # 49 frames, 6686 6734
    ["sfu_cooking_002_5"]="cam04 lyra 0 48"  # 49 frames, 4291 4339
    ["sfu_cooking_002_7"]="cam05 lyra 0 48"  # 49 frames, 4083 4131
    ["sfu_cooking_003_1"]="cam01 lyra 0 48"  # 49 frames, 7158 7206
    ["sfu_cooking_003_3"]="cam02 lyra 0 48"  # 49 frames, 5775 5823
    ["sfu_cooking_003_5"]="cam02 lyra 0 48"  # 49 frames, 3948 3996
    ["sfu_cooking_005_1"]="cam01 lyra 0 48"  # 49 frames, 5178 5226
    ["sfu_cooking_005_2"]="cam04 lyra 0 48"  # 49 frames, 4236 4284
    ["sfu_cooking_005_4"]="cam04 lyra 0 48"  # 49 frames, 3414 3462
    ["sfu_cooking_005_6"]="cam04 lyra 0 48"  # 49 frames, 2848 2896
    ["sfu_cooking_006_1"]="cam04 lyra 0 48"  # 49 frames, 13051 13099
    ["sfu_cooking_009_1"]="cam05 lyra 0 48"  # 49 frames, 5101 5149
    ["sfu_cooking_009_3"]="cam05 lyra 0 48"  # 49 frames, 6059 6107
    ["sfu_cooking_009_5"]="cam01 lyra 0 48"  # 49 frames, 4816 4864
    ["sfu_cooking_009_7"]="cam05 lyra 0 48"  # 49 frames, 3486 3534
    ["sfu_cooking_011_1"]="cam05 lyra 0 48"  # 49 frames, 7765 7813
    ["sfu_cooking_011_5"]="cam04 lyra 0 48"  # 49 frames, 2372 2420
    ["sfu_cooking_013_1"]="cam01 lyra 0 48"  # 49 frames, 4467 4515
    ["sfu_cooking_013_3"]="cam01 lyra 0 48"  # 49 frames, 6679 6727
    ["uniandes_cooking_001_10"]="cam02 lyra 0 48"  # 49 frames, 7530 7578
    ["uniandes_cooking_001_12"]="cam01 lyra 0 48"  # 49 frames, 7096 7144
    ["uniandes_cooking_002_10"]="cam01 lyra 0 48"  # 49 frames, 12040 12088
    ["uniandes_cooking_002_12"]="cam01 lyra 0 48"  # 49 frames, 4716 4764
    ["uniandes_cooking_002_6"]="cam04 lyra 0 48"  # 49 frames, 4994 5042
    ["uniandes_cooking_002_8"]="cam01 lyra 0 48"  # 49 frames, 10681 10729
    ["uniandes_cooking_003_13"]="cam01 lyra 0 48"  # 49 frames, 5911 5959
    ["uniandes_cooking_003_6"]="cam02 lyra 0 48"  # 49 frames, 5085 5133
    ["uniandes_cooking_003_8"]="cam01 lyra 0 48"  # 49 frames, 5010 5058
    ["uniandes_cooking_005_2"]="cam01 lyra 0 48"  # 49 frames, 3643 3691
    ["uniandes_cooking_005_4"]="cam04 lyra 0 48"  # 49 frames, 5743 5791
    ["uniandes_cooking_005_6"]="cam02 lyra 0 48"  # 49 frames, 4667 4715
    ["uniandes_cooking_005_8"]="cam02 lyra 0 48"  # 49 frames, 13204 13252
    ["uniandes_cooking_006_10"]="cam01 lyra 0 48"  # 49 frames, 8586 8634
    ["uniandes_cooking_006_2"]="cam01 lyra 0 48"  # 49 frames, 7977 8025
    ["uniandes_cooking_006_4"]="cam02 lyra 0 48"  # 49 frames, 3799 3847
    ["uniandes_cooking_007_2"]="cam01 lyra 0 48"  # 49 frames, 3979 4027
    ["uniandes_cooking_007_4"]="cam01 lyra 0 48"  # 49 frames, 18808 18856
    ["uniandes_cooking_007_6"]="cam01 lyra 0 48"  # 49 frames, 9611 9659
    ["uniandes_cooking_008_10"]="cam04 lyra 0 48"  # 49 frames, 3847 3895
    ["uniandes_cooking_008_4"]="cam01 lyra 0 48"  # 49 frames, 10049 10097
    ["uniandes_cooking_008_6"]="cam02 lyra 0 48"  # 49 frames, 6299 6347
    ["uniandes_cooking_008_8"]="cam02 lyra 0 48"  # 49 frames, 12084 12132
    ["uniandes_cooking_009_2"]="cam04 lyra 0 48"  # 49 frames, 10027 10075
    ["uniandes_cooking_009_4"]="cam04 lyra 0 48"  # 49 frames, 14668 14716
    ["uniandes_cooking_009_6"]="cam01 lyra 0 48"  # 49 frames, 10837 10885
    ["upenn_0628_Cooking_1_2"]="gp01 lyra 0 48"  # 49 frames, 8065 8113
    ["upenn_0702_Cooking_2_2"]="gp03 lyra 0 48"  # 49 frames, 22722 22770
    ["upenn_0702_Cooking_2_3"]="gp05 lyra 0 48"  # 49 frames, 18525 18573
    ["upenn_0702_Cooking_3_2"]="gp05 lyra 0 48"  # 49 frames, 6039 6087
    ["upenn_0702_Cooking_3_3"]="gp05 lyra 0 48"  # 49 frames, 6501 6549
    ["upenn_0702_Cooking_4_2"]="gp05 lyra 0 48"  # 49 frames, 13244 13292
    ["upenn_0702_Cooking_4_3"]="gp03 lyra 0 48"  # 49 frames, 7441 7489
    ["upenn_0702_Cooking_5_2"]="gp05 lyra 0 48"  # 49 frames, 13140 13188
    ["upenn_0702_Cooking_5_3"]="gp03 lyra 0 48"  # 49 frames, 5775 5823
    ["upenn_0710_Cooking_1_2"]="gp03 lyra 0 48"  # 49 frames, 20647 20695
    #["upenn_0710_Cooking_1_3"]="gp02 lyra 0 48"  # 49 frames, 15528 15576
    ["upenn_0710_Cooking_2_2"]="gp05 lyra 0 48"  # 49 frames, 7271 7319
    ["upenn_0710_Cooking_2_3"]="gp05 lyra 0 48"  # 49 frames, 5733 5781
    ["upenn_0710_Cooking_4_2"]="gp03 lyra 0 48"  # 49 frames, 6261 6309
    ["upenn_0710_Cooking_4_3"]="gp04 lyra 0 48"  # 49 frames, 5029 5077
    #["upenn_0712_Cooking_1_2"]="gp03 lyra 0 48"  # 49 frames, 15912 15960
    ["upenn_0712_Cooking_1_3"]="gp02 lyra 0 48"  # 49 frames, 14802 14850
    ["upenn_0712_Cooking_1_4"]="gp03 lyra 0 48"  # 49 frames, 6024 6072
    ["upenn_0712_Cooking_1_5"]="gp05 lyra 0 48"  # 49 frames, 6626 6674
    ["upenn_0712_Cooking_4_2"]="gp05 lyra 0 48"  # 49 frames, 26926 26974
    #["upenn_0712_Cooking_4_3"]="gp02 lyra 0 48"  # 49 frames, 14191 14239
    ["upenn_0712_Cooking_5_2"]="gp03 lyra 0 48"  # 49 frames, 4585 4633
    ["upenn_0712_Cooking_5_3"]="gp03 lyra 0 48"  # 49 frames, 8116 8164
    ["upenn_0714_Cooking_1_2"]="gp03 lyra 0 48"  # 49 frames, 19416 19464
    ["upenn_0714_Cooking_1_3"]="gp06 lyra 0 48"  # 49 frames, 10288 10336
    ["upenn_0714_Cooking_1_4"]="gp03 lyra 0 48"  # 49 frames, 9656 9704
    ["upenn_0714_Cooking_1_5"]="gp03 lyra 0 48"  # 49 frames, 6556 6604
    ["upenn_0714_Cooking_2_3"]="gp03 lyra 0 48"  # 49 frames, 26913 26961
    ["upenn_0714_Cooking_3_2"]="gp03 lyra 0 48"  # 49 frames, 10554 10602
    ["upenn_0714_Cooking_3_3"]="gp03 lyra 0 48"  # 49 frames, 10054 10102
    ["upenn_0714_Cooking_5_2"]="gp03 lyra 0 48"  # 49 frames, 15589 15637
    ["upenn_0714_Cooking_5_3"]="gp04 lyra 0 48"  # 49 frames, 6127 6175
    ["upenn_0714_Cooking_6_2"]="gp03 lyra 0 48"  # 49 frames, 25930 25978
    ["upenn_0714_Cooking_7_2"]="gp05 lyra 0 48"  # 49 frames, 5641 5689
    ["upenn_0714_Cooking_7_3"]="gp05 lyra 0 48"  # 49 frames, 9769 9817
)

# 공통 설정
BASE_PATH="/home/nas_main/taewoongkang/dohyeon/Exo-to-Ego/Ego4D_4DNeX_dataset/clip_crop16"

# 추론 실행 함수 (특정 GPU에서 실행)
run_inference() {
    local sequence=$1
    local camera=$2
    local pipeline=$3
    local start_frame=$4
    local end_frame=$5
    local gpu_id=$6
    
    local video_path="${BASE_PATH}/${sequence}/frame_aligned_videos/downscaled/448/${camera}.mp4"
    
    echo "[GPU $gpu_id] =========================================="
    echo "[GPU $gpu_id] Processing: ${sequence}/${camera}"
    echo "[GPU $gpu_id] Video path: $video_path"
    echo "[GPU $gpu_id] Frame range: $start_frame to $end_frame"
    echo "[GPU $gpu_id] Pipeline: $pipeline"
    echo "[GPU $gpu_id] =========================================="
    
    # 비디오 파일 존재 여부 확인
    if [[ ! -f "$video_path" ]]; then
        echo "[GPU $gpu_id] ERROR: Video file not found: $video_path"
        return 1
    fi
    
    # 특정 GPU에서만 실행되도록 CUDA_VISIBLE_DEVICES 설정
    CUDA_VISIBLE_DEVICES=$gpu_id vipe infer "$video_path" \
        --start_frame $start_frame \
        --end_frame $end_frame \
        --assume_fixed_camera_pose \
        --pipeline $pipeline
        
    echo "[GPU $gpu_id] Completed: ${sequence}/${camera}"
    echo ""
}

# 백그라운드 프로세스 관리를 위한 배열
declare -a BACKGROUND_PIDS=()
declare -a RUNNING_JOBS=()

# 실험 목록을 배열로 변환 (순서 유지를 위해)
experiment_list=()
for exp_name in "${!experiments[@]}"; do
    experiment_list+=("$exp_name")
done

# 변수 초기화
gpu_index=0
experiment_count=0
total_experiments=${#experiment_list[@]}
completed_experiments=0
failed_experiments=0

echo "Starting controlled parallel execution with max $MAX_CONCURRENT_JOBS concurrent processes"

# 실험을 순차적으로 처리하되, 동시 실행 수를 제한
for exp_name in "${experiment_list[@]}"; do
    # 현재 실행 중인 프로세스 수가 최대값에 도달했으면 대기
    while [ ${#RUNNING_JOBS[@]} -ge $MAX_CONCURRENT_JOBS ]; do
        echo "Waiting for running processes to complete... (${#RUNNING_JOBS[@]}/$MAX_CONCURRENT_JOBS slots used)"
        
        # 완료된 프로세스 확인 및 정리
        new_running_jobs=()
        for i in "${!RUNNING_JOBS[@]}"; do
            pid=${RUNNING_JOBS[$i]}
            if kill -0 $pid 2>/dev/null; then
                # 프로세스가 아직 실행 중
                new_running_jobs+=($pid)
            else
                # 프로세스가 완료됨
                if wait $pid; then
                    completed_experiments=$((completed_experiments + 1))
                    echo "Process $pid completed successfully [$completed_experiments processed]"
                else
                    failed_experiments=$((failed_experiments + 1))
                    echo "Process $pid failed [$failed_experiments failed]"
                fi
            fi
        done
        RUNNING_JOBS=("${new_running_jobs[@]}")
        
        # 짧은 대기
        sleep 1
    done
    
    # 공백으로 분리해서 파라미터 추출
    IFS=' ' read -r camera pipeline start_frame end_frame <<< "${experiments[$exp_name]}"
    
    # 현재 GPU 선택 (라운드 로빈 방식)
    current_gpu=${GPU_ARRAY[$gpu_index]}
    
    experiment_count=$((experiment_count + 1))
    echo "[$experiment_count/$total_experiments] Starting experiment $exp_name on GPU $current_gpu (slot ${#RUNNING_JOBS[@]}/$MAX_CONCURRENT_JOBS)"
    
    # 백그라운드에서 실험 실행
    run_inference "$exp_name" "$camera" "$pipeline" "$start_frame" "$end_frame" "$current_gpu" &
    
    # 백그라운드 프로세스 PID 저장
    new_pid=$!
    BACKGROUND_PIDS+=($new_pid)
    RUNNING_JOBS+=($new_pid)
    
    # 다음 GPU로 이동 (라운드 로빈)
    gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))
    
    # GPU 메모리 안정화를 위한 지연 (더 긴 대기시간)
    sleep 5
done

echo "All experiments submitted. Waiting for remaining processes to complete..."

# 남은 모든 프로세스가 완료될 때까지 대기
while [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
    echo "Waiting for ${#RUNNING_JOBS[@]} remaining processes..."
    
    new_running_jobs=()
    for i in "${!RUNNING_JOBS[@]}"; do
        pid=${RUNNING_JOBS[$i]}
        if kill -0 $pid 2>/dev/null; then
            # 프로세스가 아직 실행 중
            new_running_jobs+=($pid)
        else
            # 프로세스가 완료됨
            if wait $pid; then
                completed_experiments=$((completed_experiments + 1))
                echo "Process $pid completed successfully [$completed_experiments/$total_experiments completed]"
            else
                failed_experiments=$((failed_experiments + 1))
                echo "Process $pid failed [$failed_experiments failed so far]"
            fi
        fi
    done
    RUNNING_JOBS=("${new_running_jobs[@]}")
    
    sleep 2
done

echo "=========================================="
echo "All inference experiments completed!"
echo "Total experiments: $total_experiments"
echo "Completed successfully: $completed_experiments"
echo "Failed: $failed_experiments"
echo "Success rate: $(( completed_experiments * 100 / total_experiments ))%"
echo "=========================================="
