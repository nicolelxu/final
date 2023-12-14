# final project - tumor dataset classifier
## 📊 데이터 설명
라벨 : *'glioma_tumor'*, *'meningioma_tumor'*, *'no_tumor'*, *'pituitary_tumor'*  
라벨별 데이터 개수 : 'glioma_tumor' 826개, 'meningioma_tumor' 822개, 'no_tumor' 395개, 'pituitary_tumor' 827개  
-> 'no_tumor'의 데이터 수만 현저하게 적어 라벨별 데이터 수의 균형을 맞추기 위해 data augmentation이 필요하다.
## 원래 코드 설명
### *train_test_split의 최적의 random_state 찾기*
아래의 코드를 이용해 0부터 99까지의 정수 중 데이터 수가 현저히 적은 'no_tumor'를 제외한 나머지 데이터들의 수가 가장 균등하게 나오는 수를 찾는다. 
```
def get_labels_diff() :
    zero = 0
    one = 0
    two = 0
    thr = 0
    for i in range(len(y_train)):
        if y_train[i] == labels[0] :
            zero += 1
        elif y_train[i] == labels[1] :
            one += 1
        elif y_train[i] == labels[2] :
            two += 1
        elif y_train[i] == labels[3] :
            thr += 1
    l_list = [zero, one, thr]
    l_list = np.array(l_list)
    return np.max(l_list) - np.min(l_list)        

min_random_state_value = 100
min_random_state = 0
for i in range(100) :
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=i)
    cur_random_state_value = get_labels_diff()
    if min_random_state_value > cur_random_state_value :
        min_random_state_value = cur_random_state_value
        min_random_state = i
    if min_random_state_value == 0 :
        break

print(min_random_state_value, min_random_state)
```
'no_tumor'를 제외한 나머지 세 라벨에서 가장 데이터 수가 많은 것과 적은 것의 차이를 구한다. 69일 때 가장 작게 나왔다.  
### *data augmentation*
- blur : 'no_tumor' 데이터만 blur 처리 후 blurred_X에 저장한다
- rotate : 데이터 전체를 -20도에서 20도 사이로 랜덤하게 기울인 후 rotated_X에 저장한다.
- concatenate : train_X에 blurred_X와 rotated_X를 추가한다.
### *classification*  
<p align="center">
  <img width="574" alt="스크린샷 2023-12-13 오후 8 25 49" src="https://github.com/nicolelxu/final/assets/147023082/7b843df2-940e-4030-9d2b-ec5905866fee">  
</p>
scikit-learn의 여러 패키지 중 실험 결과 가장 accuracy가 높았던 HistGradientBoostingClassifier를 사용했다.

## 최종 코드 설명
실제 제출했을 때 위의 코드보다 train_test_split의 random_state을 0으로, data augmentation에서 blur를 'meningioma_tumor', 'no_tumor'에 처리한 것이 제출했을 때 점수가 가장 높게 나와서 해당 코드를 최종으로 했다.  
## 👩‍💻 Author
중앙대 AI학과 23학번 유금헌
## 📍 License
Licensed under the MIT License, Copyright (c) 2023 유금헌
