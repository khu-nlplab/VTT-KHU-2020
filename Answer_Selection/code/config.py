config = {
    'extractor_batch_size': 32, 
    'model_name': 'ensemble',
    'log_path': 'data/log',
    'tokenizer': 'nonword', # 'nltk' # need to check
    'batch_sizes':  (16, 24, 12),
    'lower': True,
    'use_inputs':['que','answers','subtitle','speaker','images','sample_visual','filtered_visual','filtered_sub','filtered_speaker','filtered_image','que_len','ans_len','sub_len','filtered_visual_len','filtered_sub_len','filtered_image_len', 'filtered_person_full', 'filtered_person_full_len', 'q_level_logic'],
    'stream_type': ['script', "visual_image", 'visual_bb', 'visual_meta'], # "visual_image", 'visual_bb', 'visual_meta'
    'cache_image_vectors': True,
    'image_path': 'data/AnotherMissOh/AnotherMissOh_images',
    'visual_path': 'data/AnotherMissOh/AnotherMissOh_Visual.json',
    'data_path': 'data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_set_script.jsonl',
    'subtitle_path': 'data/AnotherMissOh/AnotherMissOh_script.json',
    'glove_path': "data/glove.840B.300d.txt", # "data/glove.6B.300d.txt", "data/glove.6B.50d.txt"
    'vocab_path': "data/AnotherMissOh/vocab.pickle",
    'blackbox_answers_path' : "data/AnotherMissOh/AnotherMissOh_QA/msm_train_answers.json",
    'val_type': 'all', #  'all' | 'ch_only'
    'max_epochs': 30,
    'num_workers': 0, # --shm-size 설정 안했을시 0
    'image_dim': 512,  # hardcoded for ResNet18
    'n_dim': 300,  
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 1e-5,
    'weight_decay': 1e-5,
    'loss_name':  'knowledge_distillation_oracle_loss', #'confident_oracle_loss', # cross_entropy_loss # independent_ensemble_loss
    'optimizer': 'adam',
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': 'data/ckpt/blackbox_models',
    'ckpt_name': None,
    'max_sentence_len': 30,
    'max_sub_len': 300,
    'max_image_len': 100,
    'shuffle': (False, False, False),
    'blackbox_distillation' : False,
    #'input' : None,
    'input' :  {
        "qid": 6442,
        "shot_contained": [
            60,
            69
        ],
        "vid": "AnotherMissOh12_003_0000",
        "videoType": "scene",
        "q_level_mem": 3,
        "que": "What is Haeyoung1 drinking?",
        "q_level_logic": 3,
        "answers": [
            "Haeyoung1 is drinking a glass of water.",
            "Haeyoung1 is drinking a bottle of beer.",
            "Haeyoung1 is drinking a glass of orange juice.",
            "Haeyoung1 is drinking a glass of lemonade.",
            "Haeyoung1 is drinking a bowl of homemade wine."
        ],
        "model_1_output" : "",
        "model_2_output" : "",
    },
    "print_output" : True,
    "beta" : 100.0,
    "tau" : 0.9,
    "kd" : False,
    "kd_method" : "avg"
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'beta',
    'loss_name',
    'model_name',
    'tau',
    'kd_method',
]
