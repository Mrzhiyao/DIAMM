from query_pods import query_pods_info
from test_information import get_pod_mount_info


def get_drop_weight(tasks_type):

    result = query_pods_info()
    # print(result)
    maybe_model_list = []
    nfs_drop_local_model_weight_list = []
    local_drop_local_model_weight_list = []


    for sub in result:
        model_type_result = sub['pod'].split("-", 1)[0]  # 分割一次，取前半部分
        if model_type_result in tasks_type:
            # print(sub)
            maybe_model_list.append(sub)

    # print('maybe_model_list', maybe_model_list)
    for submodel in maybe_model_list:
        # print('submodel', submodel)
        mount = get_pod_mount_info(submodel['pod'], namespace="vllm-k8s-operator-system")
        # print(mount)
        if mount == 'API 错误: Not Found':
            # k8s查询慢，查到了已经drop的pods信息了
            continue
        try:
            mount_type = mount['volumes'][0]['type']
            nfs_drop_local_model_weight_list.append(submodel)
        except:
            mount_type = 'local'
            local_drop_local_model_weight_list.append(submodel)
        # print(mount_type)

    return nfs_drop_local_model_weight_list, local_drop_local_model_weight_list


