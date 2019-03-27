# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

position_map = {
                            '0': 'deliveryCompany',
                            '1': 'deliveryCustomerName',
                            '2': 'deliveryCustomerPhone',
                            '3': 'deliveryCustomerAddress',
                            '4': 'clientCode',
                            '5': 'notificationMode',
                            '6': 'receiveCompany',
                            '7': 'receiveCustomerName',
                            '8': 'receiveCustomerPhone',
                            '9': 'receiveCustomerAddress',
                            '10': 'goodsName',
                            '11': 'insuranceAmount',
                            '12': 'sameDayRefund',
                            '13': 'threeDaysRefund',
                            '14': 'codAmount',
                            '15': 'accountName',
                            '16': 'collectionAccount',
                            '17': 'packageFeeCanvas',
                            '18': 'deliveryInboundFee',
                            '19': 'original',
                            '20': 'fax',
                            '21': 'other'
        }
#





def gen_163():
    df = pd.read_csv('../csv_data/all_content.csv', sep=',')
    df_163 = df[df['is_163'] == 1]
    df_163_new = df_163.loc[:, ['category', 'x', 'y','w','h','length','has_PCD','class_name', 'x_y',
                                'w_h','digital_percent','digital_num','youxiangongsi_num','shengshixianqujiedao_num']]
    print(df_163_new.head(3))

    b = set(df_163_new.loc[:, 'category'])

    key_map_163 = {
        6: 0,
        7: 1,
        8: 2,
        9: 3,
        10: 4,
        21: 5,
    }
    df_163_new.index = range(0, len(df_163_new))
    print(df_163_new.head(3))
    for i in range(0, len(df_163_new)):
        df_163_new.loc[i, 'category'] = int(key_map_163[df_163_new.loc[i, 'category']])
        print(i)
    c = df_163_new.loc[0, 'category']
    d = set(df_163_new.loc[:, 'category'])


    df_163_new.to_csv("../csv_data/data_163_5.csv", index=False, sep=',')


def vis_163_info():
    df = pd.read_csv('../csv_data/all_content.csv', sep=',')
    youxiangongsi_num = df[df['youxiangongsi_num'] != 0]
    print(len(youxiangongsi_num))
    print(youxiangongsi_num.head(3))
#gen_163()
# vis_163_info()


def gen_164():
    df = pd.read_csv('../csv_data/all_content.csv', sep=',')
    df_164 = df[df['is_164'] == 1]
    df_164_new = df_164.loc[:,
                 ['category', 'x', 'y','w','h','length','has_PCD','class_name', 'x_y',
                                'w_h','digital_percent','digital_num','youxiangongsi_num','shengshixianqujiedao_num']]
    print(df_164_new.head(3))
    b = set(df_164_new.loc[:, 'category'])

    key_map_164 = {
        11: 0,
        12: 1,
        13: 2,
        14: 3,
        15: 4,
        16: 5,
        17: 6,
        18: 7,
        19: 8,
        20: 9,
        21: 10,
    }
    df_164_new.index = range(0, len(df_164_new))
    print(df_164_new.head(3))
    for i in range(0, len(df_164_new)):
        df_164_new.loc[i, 'category'] = int(key_map_164[df_164_new.loc[i, 'category']])
        print(i)
    # 加上我fake出的新数据
    df_fake164 = pd.read_csv('../csv_data/fake_164_add_dimension.csv', sep=',')
    df_fake164 = df_fake164[df_fake164['is_164'] == 1]
    df_164_choose = df_fake164.loc[:,
                 ['category', 'x', 'y','w','h','length','has_PCD','class_name', 'x_y',
                                'w_h','digital_percent','digital_num','youxiangongsi_num','shengshixianqujiedao_num']]
    df_164_choose.index = range(0, len(df_164_choose))
    print(df_164_choose.head(3))
    for i in range(0, len(df_164_choose)):
        df_164_choose.loc[i, 'category'] = int(key_map_164[df_164_choose.loc[i, 'category']])
        if i%100 == 0:
            print(i)
    res = pd.concat([df_164_new, df_164_choose], axis=0, ignore_index=True)
    res.to_csv("../csv_data/data_164_1.csv", index=False, sep=',')


def gen_161():
    df = pd.read_csv('../csv_data/all_content.csv', sep=',')
    df_161 = df[df['is_161'] == 1]
    df_161_new = df_161.loc[:,
                 ['category', 'x', 'y','w','h','length','has_PCD','class_name', 'x_y',
                                'w_h','digital_percent','digital_num','youxiangongsi_num','shengshixianqujiedao_num']]
    a = df_161_new.loc[0, 'category']
    b = set(df_161_new.loc[:, 'category'])

    key_map_161 = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        5: 4,
        4: 5,
        21: 6,
    }
    for i in range(0, len(df_161_new)):
        df_161_new.loc[i, 'category'] = int(key_map_161[df_161_new.loc[i, 'category']])
        print(i)
    c = df_161_new.loc[0, 'category']
    d = set(df_161_new.loc[:, 'category'])

    df_161_new.to_csv("../csv_data/data_161_1.csv", index=False, sep=',')
gen_161()
def gen_164_2():
    df = pd.read_csv('../csv_data/data_164.csv',sep=',')
    print(df.head(3))
    df_insuranceAmount = df[df['category']==0]
    df_other = df[df['category']!=0]
    print(df_insuranceAmount.head(3))
    df_insuranceAmount.index = range(0, len(df_insuranceAmount))
    for i in range(0, len(df_insuranceAmount)):
        a = 0.05
        b = df_insuranceAmount.loc[i, 'x']
        df_insuranceAmount.loc[i, 'x'] = a + (b-a)*np.random.random()
        print(i)
    print(df_insuranceAmount.head(3))
    print(len(df_insuranceAmount))
    print(len(df_other))
    df_new = df_insuranceAmount.append(df_other)
    print(len(df_new))
    c = df_new.loc[0, 'category']
    d = set(df_new.loc[:, 'category'])
    df_new.to_csv("../csv_data/data_164_new.csv", index=False, sep=',')


# -----------------------------------
#

# -----------------------
#
def get_content():
    import json
    with open('../csv_data/block1.txt', 'r') as f:
        pos_lines = f.readlines()

    df = pd.read_csv('../csv_data/result_block1.csv')
    print('there are line:', len(df))
    d = set(df.loc[:, 'category'])
    print d
    for idx, pos in enumerate(pos_lines):
        a = pos.strip()
        a_dict = json.loads(a)
        text = a_dict['text']
        cls_feature = df.loc[idx]
        # print('text:', text)
        if cls_feature['category'] == 'undefined':
            print(text)
            print(cls_feature)
            df.drop(df.index[[idx]],inplace=True)
            print('drop one line,remain ', len(df))
            continue
        else:
            assert df.loc[idx, 'length']*10 == len(text)
            df.loc[idx, 'content'] = text.encode('utf-8')

    d = set(df.loc[:, 'category'])
    print d

    df.to_csv('../csv_data/result_content.csv', index=False, sep=',')


def vis_164():
    df = pd.read_csv('../csv_data/result_content.csv')
    df_164 = df[df['is_164']==1]
    df_164.to_csv('../csv_data/vis_164.csv', index=False, sep=',')


def vis_164_insuranceAccount():

    df_164 = pd.read_csv('../csv_data/vis_164.csv')
    cate = set(df_164.loc[:, 'category'])
    print(cate)
    df_11 = df_164[df_164['category'] == 11]
    content = df_11['content']
    df_11.to_csv('../csv_data/insuranceAcount.csv', index=False, sep=',')

def gen_164_3():
    """
    将高宽两个特征去除，试试
    :return:
    """
    df = pd.read_csv('../csv_data/data_164_new.csv',sep=',')
    print(df.head(3))
    df_insuranceAmount = df[df['category']==0]
    df_other = df[df['category']!=0]
    print(df_insuranceAmount.head(3))
    df_insuranceAmount.index = range(0, len(df_insuranceAmount))
    for i in range(0, len(df_insuranceAmount)):
        a = 0
        b = df_insuranceAmount.loc[i, 'x']
        df_insuranceAmount.loc[i, 'x'] = a + (b-a)*np.random.random()
        print(i)
    print(df_insuranceAmount.head(3))
    print(len(df_insuranceAmount))
    print(len(df_other))
    df_new = df_insuranceAmount.append(df_other)
    print(len(df_new))
    c = df_new.loc[0, 'category']
    d = set(df_new.loc[:, 'category'])
    df_new = df_new.loc[:, ['category','x','y','length','is_digital','has_PCD','class_name','class_prob']]
    df_new.to_csv("../csv_data/data_164_new2.csv", index=False, sep=',')

def re_gen_163():
    """
    对163的训练数据短文本分类得分这一特征去掉
    :return:
    """
    df = pd.read_csv('../csv_data/data_163.csv', sep=',')
    print(df.head(3))
    df_remove_score = df.loc[:, ['category','x','y','w','h','length','is_digital','has_PCD','class_name']]
    print(df_remove_score.head(3))
    df_remove_score.to_csv('../csv_data/data_163_1.csv', index = False, sep=',')

def re_gen_163_2():
    """
    对163的训练数据短文本分类得分这一特征去掉
    :return:
    """
    df = pd.read_csv('../csv_data/data_163.csv', sep=',')
    print(df.head(3))
    df_remove_score = df.loc[:, ['category','x','y','w','h','length','is_digital','has_PCD']]
    print(df_remove_score.head(3))
    df_remove_score.to_csv('../csv_data/data_163_2.csv', index = False, sep=',')

def re_gen_163_3():
    """
    对163的训练数据短文本分类w这一特征去掉
    :return:
    """
    df = pd.read_csv('../csv_data/data_163.csv', sep=',')
    print(df.head(3))
    df_remove_score = df.loc[:, ['category','x','y','h','length','is_digital','has_PCD', 'class_name']]
    print(df_remove_score.head(3))
    df_remove_score.to_csv('../csv_data/data_163_3.csv', index=False, sep=',')


def get_digital_info(content):
    """
    通过文本内容得到其中的数字个数和数字占百分比
    :param content:
    :return:
    """
    content = str(content)
    content = content.decode('utf-8')
    int_num = 0
    for _ in content:
        if _.isdigit():
            int_num+=1
    char_num = len(content)
    int_percent = int_num*1./char_num
    return int_percent, int_num


def get_has_youxingongsi_num(content):
    """
    根据内容确定是不是含有有限公司的字样
    :param content:
    :return:
    """
    content = str(content)
    content = content.decode('utf-8')
    care_list = [u'有', u'限', u'公', u'司']
    youxiangongsi_num = 0
    for _ in content:
        if _ in care_list:
            youxiangongsi_num+=1
    return youxiangongsi_num


def get_has_shengshixianqujiedao_num(content):
    """
    根据内容，给出是否拥有地址关键字的数目
    :param content:
    :return:
    """
    content = str(content)
    content = content.decode('utf-8')
    care_list = ['省', '市', '自治区', '市辖区', '市', '自治州', '地区', '盟', '区', '县', '自治县', '林区', '旗', '自治旗', '街道办事处', '街道', '办事处', '乡', '苏木', '镇', '农场', '监狱', '区公所', '居委会', '村委会', '嘎查', '社区']
    care_list = [_.decode('utf-8') for _ in care_list]
    shengshixianqujiedao_num = 0
    for _ in care_list:
        if _ in content:
            shengshixianqujiedao_num+=1
    return shengshixianqujiedao_num
def gen_new_dimension():
    """
    数字百分比，
    x/y
    w/h
    digital_percent
    digital_num
    english_percent
    youxiangongsi_num
    shengshixianqujiedao_num
    :return:
    """
    df = pd.read_csv('../csv_data/result_content.csv', sep=',')
    df['x_y'] = None
    df['w_h'] = None
    df['digital_percent'] = None
    df['digital_num'] = None
    df['youxiangongsi_num'] = None
    df['shengshixianqujiedao_num'] = None
    for i in range(0, len(df)):
        df.loc[i, 'x_y'] = float(df.loc[i, 'x'])/(float(df.loc[i, 'y'])+1)
        df.loc[i, 'w_h'] = float(df.loc[i, 'w'])/float(df.loc[i, 'h'])
        digital_percent, digital_num = get_digital_info(df.loc[i, 'content'])
        df.loc[i, 'digital_percent'] = digital_percent
        df.loc[i, 'digital_num'] = digital_num
        youxiangongsi_num = get_has_youxingongsi_num(df.loc[i, 'content'])
        df.loc[i, 'youxiangongsi_num'] = youxiangongsi_num
        shengshixianqujiedao_num = get_has_shengshixianqujiedao_num(df.loc[i, 'content'])
        df.loc[i, 'shengshixianqujiedao_num'] = shengshixianqujiedao_num
        #print(df.head(3))
        if i %100 == 0:
            print('finish'+str(i))
    df.to_csv('../csv_data/all_content.csv', index=False, sep=',')

def re_make_fake_data():
    """
    给我fake出的164数据增加特征维度，
    fake_164.csv
    :return:
    """
    df = pd.read_csv('../csv_data/fake_164.csv', sep=',')
    df['x_y'] = None
    df['w_h'] = None
    df['digital_percent'] = None
    df['digital_num'] = None
    df['youxiangongsi_num'] = None
    df['shengshixianqujiedao_num'] = None
    for i in range(0, len(df)):
        df.loc[i, 'x_y'] = float(df.loc[i, 'x'])/(float(df.loc[i, 'y'])+1)
        df.loc[i, 'w_h'] = float(df.loc[i, 'w'])/float(df.loc[i, 'h'])
        digital_percent, digital_num = get_digital_info(df.loc[i, 'content'])
        df.loc[i, 'digital_percent'] = digital_percent
        df.loc[i, 'digital_num'] = digital_num
        youxiangongsi_num = get_has_youxingongsi_num(df.loc[i, 'content'])
        df.loc[i, 'youxiangongsi_num'] = youxiangongsi_num
        shengshixianqujiedao_num = get_has_shengshixianqujiedao_num(df.loc[i, 'content'])
        df.loc[i, 'shengshixianqujiedao_num'] = shengshixianqujiedao_num
        #print(df.head(3))
        if i %100 == 0:
            print('finish'+str(i))
    df.to_csv('../csv_data/fake_164_add_dimension.csv', index=False, sep=',')
#re_make_fake_data()