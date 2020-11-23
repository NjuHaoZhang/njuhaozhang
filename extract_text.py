import sys
import re
import json

def extract_fiction_text():
    input_file = './fiction_info.txt'
    ouput_file = './fiction_text.txt'

    cnt = cnt_good = cnt_bad = 0
    used = set()
    with open(input_file, 'r') as fr, open(ouput_file, 'w') as fs:
        for line in fr:
            datas = line.strip().split('\t')
            if line in used:
                print(line.strip())
                continue
            cnt += 1
            if cnt == 1:
                print(datas)
                fs.write('id\tcategory\ttags\ttitle\tsummary\tideo_url\n')
                continue
            if len(datas) != 17:
                cnt_bad += 1
                continue
            category, category_c, original_url, play_length, tags, tags_c, video_id, original_title, original_title_c, \
                    title, title_c, summary, summary_c, height, width, image_size, video_url = datas[:]

            ids = video_url.split('rawKey=')[-1]
            category_c = category_c.strip('"')
            tags_c = tags_c.strip('"')
            original_title_c = original_title_c.strip('"')
            title_c = title_c.strip('"')
            summary_c = summary_c.strip('"')
            
            # print(category_c, tags_c, original_title_c, title_c, summary_c)
            # fs.write('{}\t{}\t{}\n'.format(cnt, json.dumps([category_c, tags_c, title_c, summary_c], ensure_ascii = False), video_url))
            fs.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(ids, category_c, tags_c, title_c, summary_c, video_url))
            used.add(line)

    print('total: {}, good: {}, bads: {}'.format(cnt, cnt_good, cnt_bad))


def extract_novel_text():
    input_file = './novel_info.txt'
    ouput_file = './novel_text.txt'

    cnt = cnt_good = 0
    pattern = re.compile('<[^>]+>')
    used = set()
    with open(input_file, 'r') as fr, open(ouput_file, 'w') as fs:
        for line in fr:
            datas = line.strip().split('\t')
            if line in used:
                print(line.strip())
                continue
            cnt += 1
            if cnt == 1:
                print(datas)
                fs.write('id\tcategory\ttags\ttitle\tcontent\n')
                continue
            ids, cps, create_time, data, item_type, seed_name, update_time, p_date = datas[:]
            info = json.loads(data)
            original_title = info['original_title']
            title = info['title']
            category = ','.join(info['category'])
            tags = ','.join(info['tags'])
            content = pattern.sub("", info['content'])
            normalized_content = info['normalized_content']
            fs.write('{}\t{}\t{}\t{}\t{}\n'.format(ids, category, tags, title, content))
            used.add(line)
    print('total: {}'.format(cnt))

if __name__ == '__main__':
    extract_fiction_text()
    extract_novel_text()
