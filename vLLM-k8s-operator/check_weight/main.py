from check_node_weight import check_remote_dir_exists, check_remote_file_exists
import os
import time

NODES_IP = {
    'b410-4090d-1': '192.168.2.75',
    'b410-4090d-2': '192.168.2.190',
    'b410-4090d-3': '192.168.2.78',
    'b410-3090-1': '192.168.2.80',
    'b410-2070s-1': '192.168.2.5',
    'b410-2070s-2': '192.168.2.6',
    'b410-2070s-3': '192.168.2.7',
    # 'b410-2070s-4': '192.168.2.133',
}

ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
remote_full_path = '/home/yaozhi/images/'
flag = 'transfer_complete.flag'

local_weight_paths = {
    'text2text':{
                '/home/yaozhi/images/glm-4-9b-chat',
                '/home/yaozhi/images/glm4-9b-ollama',
                '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct',
                '/home/yaozhi/images/llama31-8b-ollama',
                '/home/yaozhi/images/Qwen2.5-7B-Instruct',
                '/home/yaozhi/images/Qwen2.5-7B-ollama',
                '/home/yaozhi/images/Qwen2.5-3B-Instruct',
                '/home/yaozhi/images/Qwen2.5-3B-ollama',
                },
    'image2text':{'/home/yaozhi/images/Qwen2-VL-7B-Instruct', 
                '/home/yaozhi/images/Qwen2-VL-2B-Instruct'
                },
    'text2image':{'/home/yaozhi/images/yz_diffusion'},
    'image2image':{'/home/yaozhi/images/yz_diffusion'},
    # 'text/image2image':{'/home/yaozhi/images/yz_diffusion'},
    'text2video':{'/home/yaozhi/images/CogVideo-1.0'}
}

while 1:

    for key,value in local_weight_paths.items():
        # print('key', value)
        remote_weight_paths = local_weight_paths[key]
        for remote_weight_path in remote_weight_paths:
            remote_flag_path = remote_weight_path + '_'+ flag
            # print('remote_weight_path', remote_weight_path)
            for sub_noe in NODES_IP:
                r_weight = check_remote_dir_exists(NODES_IP[sub_noe], remote_weight_path, ssh_key_path)
                r_flag = check_remote_file_exists(NODES_IP[sub_noe], remote_flag_path, ssh_key_path)

    time.sleep(30)

