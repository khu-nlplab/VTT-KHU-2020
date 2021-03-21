 Video Turing Test 2020<br>
=====
Question Level classification (QLC)
-----------------------------
입력 질문에 대한 난이도를 측정하는 모델입니다.
(Choi et al. 2020) "DramaQA: Character-Centered Video Story Understanding with Hierarchical QA" 에 정의되어 있는 Criteria 중 2가지(Memory, Logical Complexity)의 관점에서 난이도를 측정합니다. <br><br>
__How to Use__
1. ``Question Difficulty Estimation/requirement.sh``을 실행시켜주세요
> ./requirement.sh

2. 다음 링크의 미리 학습된 모델을 받아주세요<br>
모델을 다운로드 받은 후 checkpoints 디렉토리를 만들어 checkpoints 디렉토리 안에 다운 받은 모델을 넣어 주세요. <br>
question level classification pre-trained model download link : <br>
>[Google drive link](https://drive.google.com/drive/folders/1RUj_tEFbCpfPTPL0_9eA3QsbaHcymD3r?usp=sharing)

3. ``Question Difficulty Estimation/LevelClassificationModel.py``를 import하고 해당 모듈에 선언된 LevelClassificationModel class를 사용하시면 되겠습니다.
> 현재 main.py 파일을 사용예시로 해두었습니다. 

4. ``output``으로 다음 두가지가 출력됩니다.
> Memory Level : 3 <br>
> Logic Level : 4

Contact : Su-Hwan Yoon (yunsh3432@khu.ac.kr)

Multiple Answer Selection (MAS)
------------------
다중의 응답을 입력으로 받아 가장 적절한 응답을 선택하는 모델입니다. <br><br>
__How to Use__


DramaQA starter code를 기반으로 작성했으니 해당 github를 참고하면 도움됩니다.
>[DramaQA code linke](https://github.com/liveseongho/DramaQAChallenge2020)

Answer Selection을 실행하기 위해선 AnotherMissOh 데이터를 가지고 있어야합니다.
```
data/
  AnotherMissOh/
    AnotherMissOh_images/
        $IMAGE_FOLDERS
        cache
    AnotherMissOh_QA/ -> inference에선 불필요
    AnotherMissOh_Visual.json
    AnotherMissOh_script.json
  ckpt/
```

1. 다음 링크의 미리 학습된 모델과 전처리된 데이터 ``Answer_Selection/data`` 디렉토리에 압축해제 해주세요.<br>
Answer selection classification pre-trained model and data download link: <br>
>[Google drive link](https://drive.google.com/drive/folders/1H9wTPtn8fwJcmLlfJN_li52TWzit6MGT?usp=sharing)

2. ``Answer_Selection/code``에 들어가서 다음 명령어 실행.
>pip install -r requirements.txt

3. ``Answer_Selection/code/run.sh``를 실행하면 output이 출력됩니다.
>bash run.sh

4. ``output``으로 다음 두가지가 출력됩니다.
>correc_idx : int, 
>answer : str

Contact : Gyu-Min Park (pgm1219@khu.ac.kr)

만약 작년 연구 모델이 보고싶으시다면 <br>
3차년도 연구 모델 링크 : https://github.com/khu-nlplab/VTT-KHU-2019
