# 自定义解码器，尝试将字符串转换为数字
def json_decode_with_int(obj):
    if isinstance(obj, dict):
        return {k: json_decode_with_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_decode_with_int(i) for i in obj]
    elif isinstance(obj, str):
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                return obj
    return obj


# 将字典中的所有集合转换为列表
def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_sets(i) for i in obj]
    return obj