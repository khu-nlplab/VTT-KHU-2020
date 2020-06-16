from LevelClassificationModel import LevelClassificationModel
import torch


def main():
    question = "Why did Sukyung block Jinsang?"
    utterance = "Good morning. Says who?[SEP]What? Oh man... Well?[SEP]I can't say it is or isn't.[SEP]"
    answers = "Because Sukyung wanted to ask Jinsang who the woman that Dokyung was seeing is."


    model = LevelClassificationModel()
    memory_level, logic_level = model.predict(question, utterance, answers)

    print(memory_level) #gold 3
    print(logic_level) #gold 4

if __name__=="__main__":
    main()