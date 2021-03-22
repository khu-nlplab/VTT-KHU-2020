from LevelClassificationModel import LevelClassificationModel
import torch


def main():
    question = "Why did Sukyung block Jinsang?"
    vid = "AnotherMissOh01_001_0000"

    bert_config_file = 'checkpoints/bert_config.json'  # bert config file path
    vocab_file = 'checkpoints/vocab.txt'  # bert vocabulary file path
    memory_model_path = 'checkpoints/Memory_level_model.bin'  # memory prediction model file path
    logical_model_path = 'checkpoints/Logical_level_model.bin'  # logic prediction model file path 

    model = LevelClassificationModel(bert_config_file, vocab_file, memory_model_path, logical_model_path)
    memory_level, logic_level = model.predict(question, vid)

    print(memory_level) #gold 3
    print(logic_level) #gold 4

if __name__=="__main__":
    main()
