scp -r C:\ai\deepLearning\mylora\modelscope yc@172.29.7.140:home/yc/
accelerate test --config_file ddp_config.yaml
accelerate launch --config_file ddp_config.yaml distributed_data_parallel.py
accelerate test --config_file fsdp_config.yaml
accelerate launch --config_file fsdp_config.yaml distributed_data_parallel.py