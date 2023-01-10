_base_ = [
    '../_base_/models/bisenetv2.py',    # model settings
    '../_base_/datasets/waterpuddles.py', # dataset settings
    '../_base_/schedules/schedule_40k.py', # scheduler settings
    '../_base_/default_runtime.py' # other runtime settings
]
