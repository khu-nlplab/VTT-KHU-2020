from LevelClassificationModel import LevelClassificationModel
import time

def main():

    bert_config_file = "model/bert_config.json"
    vocab_file = "model/vocab.txt"
    dropout_prob = 0.2
    memory_model_path = "model/pytorch_memory_model.bin"
    logical_model_path = "model/pytorch_logic_model.bin"

    question = "Why does Haeyoung1 stop at the radio channel that is offering some advice to the audience?"
    description = "Haeyoung1 suddenly wakes of from her bed still drunk. Haeyoung1 turns on the radio and dances to the music. While Haeyoung1 is changing the channel of the radio, Haeyoung1 happens to listen to the radio program that giving advice for the audience.Haeyoung1 is getting up from the bed. Haeyoung1 is turning on the radio and dancing. Haeyoung1 is changing the radio channel."
    utterance = "Like a refreshing club soda that will relief your gas, Like the sage of all sagesIf you'd like to receive some advice from Mr. Lee, Byeong-jun, then please give us a call right now. "

    model = LevelClassificationModel(bert_config_file, vocab_file, dropout_prob, memory_model_path, logical_model_path)
    
    start = time.time()
    memory_level, logic_level = model.predict(question, description, utterance)
    end = time.time()

    print("processing time: {}".format(end-start))
    print(memory_level) #gold 3
    print(logic_level) #gold 4

if __name__=="__main__":
    main()
