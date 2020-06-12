import json
# 追加的形式，向 json 文件中 写入新内容
def save_json(save_path, item):
    # 多process 同时写 会出错，so要加锁(process-level)
    import fcntl, threading
    # id = threading.currentThread().getName()
    if not os.path.exists(save_path):
        data_dict = {}
        data_dict[item[0]] = item[1]
        with open(save_path, "w") as fp: # 创建文件 并直接写
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
            json.dump(data_dict, fp)
    else:
        with open(save_path, "r") as fp:  # 先读出原有内容再
            data_dict = json.load(fp)
            data_dict[item[0]] = item[1]
        with open(save_path, "w+") as fp:  # 追加
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
            json.dump(data_dict, fp)

    #

def load_json(save_path, key):
    with open(save_path, "r") as fp:
        data_dict = json.load(fp)
        return data_dict[key]

# 单元测试 save_json and load_json
def test_load_save_json(save_path):
    dict = [("key-{}".format(idx),"world-{}".format(idx)) for idx in range(5)]
    print("dict: ", dict)
    for item in dict:
        print("item: ", item)
        save_json(save_path, item)
    with open(save_path, "r") as fp:
        data_dict = json.load(fp)
    keys = list(data_dict.keys())
    print("keys: ", keys)
    for key in keys:
        print(load_json(save_path, key))
