python my_projects/scripts/visualize_preds.py my_projects/test_results/hots_v1/ hots -raw -gt -sn HOTS_set
python my_projects/scripts/visualize_preds.py my_projects/test_results/hots_v1_cat/ hots-c -raw -gt -sn HOTS-C_set
python my_projects/scripts/visualize_preds.py my_projects/test_results/irl_vision/ irl_vision -raw -gt -sn SOD_set
python my_projects/scripts/visualize_preds.py my_projects/test_results/irl_vision_cat/ irl_vision_cat -raw -gt -sn SOD-C_set
python my_projects/scripts/visualize_preds.py my_projects/test_results/arid20_cat/ ARID20 -raw -gt -sn ARID20_set
python my_projects/scripts/visualize_preds.py my_projects/test_results/arid10_cat/ ARID10 -raw -gt -sn ARID10_set

python my_projects/scripts/visualize_preds.py my_projects/test_results/arid20_cat/ ARID20 -raw -gt -sn ARID20_clutter_set -cl 


python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/test_results/hots2hots_cat HOTS -raw -gt -sn HOTS2HOTS-C -ods HOTS -tds HOTS-C

python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/test_results/irl_vision2irl_vision_cat SOD -raw -gt -sn SOD2SOD-C -ods SOD -tds SOD-C



python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/test_results/hots_cat2irl_vision_cat HOTS-C -raw -gt -sn HOTS-C2SOD-C -ods HOTS-C -tds SOD-C

python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/test_results/irl_vision_cat2hots_cat SOD-C -raw -gt -sn SOD-C2HOT-C -ods SOD-C -tds HOTS-C 

python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/test_results/arid202arid10 ARID20 -raw -gt -sn ARID20_ARID10 -ods ARID20 -tds ARID10

python my_projects/scripts/visualize_preds.py my_projects/test_results/sodhots-c/ sodhots-c -raw -gt -sn SODHOTS-C_set



python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/zero_shot/qualitative_results/hots_cat ADE20K -raw -gt -sn zero_shot_HOTS-C  -ods ADE20K -tds HOTS-C

python my_projects/scripts/visualize_preds.py my_projects/conversion_tests/zero_shot/qualitative_results/irl_vision_cat ADE20K -raw -gt -sn zero_shot_SOD-C -ods ADE20K -tds SOD-C