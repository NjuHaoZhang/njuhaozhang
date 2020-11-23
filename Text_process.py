import re
import string

#
stopwords = ['的', '呀', '这', '那', '就', '的话', '如果']
#

# 去掉文本中的停用词
def drop_stopwords(content, stopwords):
    content_str = ''
    for word in content:
        if word in stopwords:
            continue
        content_str += word
    return content_str

def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str

def pre_process(input_file, flags):
    cnt = 0
    conetents = []
    with open(input_file, 'r') as fr:
        for line in fr:
            datas = line.strip().split('\t')
            cnt += 1
            if cnt == 1:
                continue
            ids = datas[0]
            texts = datas[1:] if flags == 'novel' else datas[1:-1]
            concat_text = ' '.join(texts)
            # print("texts:\n")
            # print(texts)
            # print("concat_text:")
            # print(concat_text)
            # print("\n")

            # ----- handle concat_text only ---- #

            output = format_str(concat_text)
            # print ("output after del un_related word:")
            # print(output)
            # print("\n")

            # del stop word
            output = drop_stopwords(output, stopwords)
            # print("output after drop stop word:")
            # print(output)
            # print("\n")

            # 去重
            output = list(set(output))
            output = ''.join(output)

            # to construct dictionary
            conetents.append(output)
            # print(conetents)
    # print(u'file: {}, total hand: {}'.format(input_file, cnt))

    return conetents

def cal_concur(A, B):
    nums = len(A)
    cnt = 0
    for i in range(nums):
        if A[i] in B:
            cnt += 1
    return cnt

def main():
    input_file = '/Users/haozhang/Desktop/Project/dataset/novel_text.txt'
    # ouput_file = './novel_features.txt'
    dic_novel = pre_process(input_file, 'novel')

    input_file = '/Users/haozhang/Desktop/Project/dataset/fiction_text.txt'
    # ouput_file = './fiction_features.txt'
    dic_fiction = pre_process(input_file, 'fiction')

    # distance calculation
    score = []
    nums_novel, nums_fiction = len(dic_novel), len(dic_fiction)
    for i in range(nums_novel):
        tmp_s = []
        for j in range(nums_fiction):
            tmp = cal_concur(dic_novel[i], dic_fiction[j])
            tmp_s.append(tmp)
        score.append(tmp_s)

    print(score)
    res = []
    for s in score:
        idx = s.index(max(s))
        res.append(idx)
    print("res: ", res)

    #
    # for i in range(nums_novel):
    #     novel_key = dic_novel[i]
    #     fic_key = dic_fiction[res[i]]
    #     print("novel_key", novel_key)
    #     print("fic_key: ", fic_key)

    cnt = 0
    novel = []
    input_file = '/Users/haozhang/Desktop/Project/dataset/novel_text.txt'
    with open(input_file, 'r') as fr:
        for line in fr:
            datas = line.strip().split('\t')
            cnt += 1
            if cnt == 1:
                continue
            novel_content = datas[-1]
            novel.append(novel_content)
    #
    cnt = 0
    fic = []
    input_file = '/Users/haozhang/Desktop/Project/dataset/fiction_text.txt'
    with open(input_file, 'r') as fr:
        for line in fr:
            datas = line.strip().split('\t')
            cnt += 1
            if cnt == 1:
                continue
            novel_content = datas[3]
            fic.append(novel_content)
    #
    print("res_len: ", len(res))
    print("novel_len: ", len(novel))
    print("fic_len: ", len(fic))

    print("\n\n")
    for i in range(nums_novel):
        if i == 0:
            continue
        print("novel: ", novel[i])
        print("fic: ", fic[res[i]])
        print("resi: ", res[i])
        print("\n\n")


if __name__ == "__main__":
    main()
