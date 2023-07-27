import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
import cupy as cp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-input_path', help=' : Please set the input data path', default="./data/result.csv") 
parser.add_argument('-save_path', help=' : Please set the save data path', default="./data/") 
parser.add_argument('-method', help=' : both(Doing all) or sim(Similarity Feature Selection) or cox(Cox-EN Feature Selection)')
parser.add_argument('-cox_ratio', type=float, help=' : Please set the L1 ratio of Cox-EN', default=0.005)
parser.add_argument('-sim_threshold', type=float, help=' : Please set the threshold of Similarity Feature Selection', default=0.6) 
args = parser.parse_args()

def load_data(path="./data/result.csv"):
    data = pd.read_csv(path)
    data = data.drop(['Patient Identifier'], axis=1)
    data = data.drop(data[data['OS_MONTHS']=="[Not Available]"].index)
    data = data.astype({'OS_MONTHS': 'float'})
    # 특징과 생존 기간, 생존 여부를 분리
    y_event = data['OS_EVENT']
    y_month = data['OS_MONTHS']
    features = data.drop(["OS_EVENT","OS_MONTHS"], axis=1)

    return data, features, y_event, y_month

# 데이터프레임의 각 열을 열벡터로 변환하여 Similarity 행렬 계산
def calculate_similarity_matrix_gpu(data):
    print("===========< Similarity Feature Selection >===========")
    print(f"| # of Before Columns : {data.shape[1]}")
    data_gpu = cp.asarray(data.values)
    similarity_matrix_gpu = cp.zeros((data_gpu.shape[1], data_gpu.shape[1]), dtype=cp.float32)

    # 각 열 벡터를 기준으로 나머지 열과의 Similarity 계산
    for i in tqdm(range(data_gpu.shape[1])):
        # 기준 열 벡터
        column_vector_gpu = data_gpu[:, i]
        # 나머지 열 벡터들과의 Similarity 계산
        similarity_vector_gpu = cp.dot(data_gpu[:, i+1:].T, column_vector_gpu[:, cp.newaxis])
        # Similarity 행렬에 열 벡터 추가
        similarity_matrix_gpu[i, i+1:] = similarity_vector_gpu.flatten()
    similarity_matrix = cp.asnumpy(similarity_matrix_gpu)

    return similarity_matrix

def feature_selection_similarity(similarity_matrix, y_event, y_month, threshold=0.6, size_ratio=0.2, random_seed=42):
    # similarity_matrix의 각 열들의 값들 중 60%가 0보다 작거나 같은 열 제거
    columns_to_drop = np.sum(similarity_matrix <= 0, axis=0) >= threshold * similarity_matrix.shape[0]
    # 데이터프레임에서 제거할 열 이름들 가져오기
    columns_to_drop_names = features.columns[columns_to_drop]
    filtered_features = features.drop(columns_to_drop_names, axis=1)
    filtered_features['OS_EVENT'] = y_event
    filtered_features['OS_MONTHS'] = y_month

    print(f"| # of After Columns : {filtered_features.shape[1]}")
    print(f"| # of Columns : {data.shape[1]} -> {filtered_features.shape[1]}")

    # Train, Validation, Test 데이터셋 분할
    train_dataset, test_dataset = train_test_split(filtered_features, test_size=size_ratio, random_state=random_seed)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=size_ratio, random_state=random_seed)

    print(f"| # of train_dataset : {train_dataset.shape[0]}")
    print(f"| # of valid_dataset : {val_dataset.shape[0]}")
    print(f"| # of test_dataset : {test_dataset.shape[0]}")
    print("==================================================")

    return filtered_features, train_dataset, val_dataset, test_dataset

def similarity_analysis(similarity_matrix):
    fig = plt.figure(figsize=(10, 8))  # 가로 10인치, 세로 8인치 크기의 플롯 생성
    plt.imshow(similarity_matrix, cmap='coolwarm', aspect='auto', vmin=-1000, vmax=1000)
    plt.colorbar()
    plt.xlabel('Columns')
    plt.ylabel('Columns')
    plt.title('Similarity Matrix')
    plt.show()

def save_csv(train, valid, test, path="./data/", method="similarity"):
    train.to_csv(path+"train_"+method+"_result.csv")
    valid.to_csv(path+"valid_"+method+"_result.csv")
    test.to_csv(path+"test_"+method+"_result.csv")

def feature_selection_cox_en(data, l1_ratio=0.005):
    print("===========< Cox_EN Feature Selection >===========")
    print(f"| # of Before Columns : {data.shape[1]}")
    X = data.drop(['OS_MONTHS', 'OS_EVENT'], axis=1)
    y_duration = data['OS_MONTHS']
    y_event = data['OS_EVENT']

    # Cox-EN 모델 구축
    cox_en = ElasticNetCV(l1_ratio=l1_ratio)
    cox_en.fit(X, y_duration, y_event)

    # Cox-EN을 통한 Feature Selection된 Column 추출
    col_idx = np.where(cox_en.coef_ != 0)[0]
    # 추출된 Column에 의한 데이터프레임 구성
    filtered_features = X.iloc[:, col_idx].copy()
    filtered_features['OS_MONTHS'] = y_duration
    filtered_features['OS_EVENT'] = y_event

    print(f"| # of After Columns : {filtered_features.shape[1]}")
    print(f"| # of Columns : {data.shape[1]} -> {filtered_features.shape[1]}")

    # Train, Validation, Test 데이터셋 분할
    train_dataset, test_dataset = train_test_split(filtered_features, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    print(f"| # of train_dataset : {train_dataset.shape[0]}")
    print(f"| # of valid_dataset : {val_dataset.shape[0]}")
    print(f"| # of test_dataset : {test_dataset.shape[0]}")
    print("==================================================")

    return filtered_features, train_dataset, val_dataset, test_dataset

def similarity_processing(features, threshold, path):
    # 데이터프레임 또는 배열로부터 Similarity 행렬 계산
    similarity_matrix = calculate_similarity_matrix_gpu(features)
    # 유사도 행렬 시각화
    similarity_analysis(similarity_matrix)
    # Similarity를 이용한 Feature Selection 진행 및 train_valid_test split 진행
    feature_similarity, train_dataset, val_dataset, test_dataset = \
        feature_selection_similarity(similarity_matrix, y_event, y_month, threshold=threshold)

    # csv 파일로 train, valid, test 각각 저장
    save_csv(train_dataset, val_dataset, test_dataset, path=path, method="similarity")

def cox_en_processing(data, l1_ratio, path):
    # Cox-EN을 이용한 Feature Selection 진행 및 train_valid_test split 진행
    filtered_cox_en, train_dataset, val_dataset, test_dataset = \
        feature_selection_cox_en(data, l1_ratio=l1_ratio)

    # csv 파일로 train, valid, test 각각 저장
    save_csv(train_dataset, val_dataset, test_dataset, path=path, method="cox_en")

if __name__ == '__main__' :
    argv = sys.argv
    
    # 데이터 로드(result.csv)
    data, features, y_event, y_month = load_data(path=args.input_path)

    if args.method == "both":
        similarity_processing(features, threshold=args.sim_threshold, path=args.save_path)
        cox_en_processing(data, l1_ratio=args.cox_ratio, path=args.save_path)
    elif args.method == "sim":
        similarity_processing(features, threshold=args.sim_threshold, path=args.save_path)
    elif args.method == "cox":
        cox_en_processing(data, l1_ratio=args.cox_ratio, path=args.save_path)
