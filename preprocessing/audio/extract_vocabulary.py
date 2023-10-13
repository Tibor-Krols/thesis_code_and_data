from preprocessing.audio.extract_timestamps_words_audio import extract_word_timestamps,load_textgrid
from training.train_test_split import get_train_test_sections
class Vocab:
    def __init__(self):
        self.train_sections,self.test_sections = get_train_test_sections()
        self.train_vocab = self.get_train_vocab()
        self.test_vocab = self.get_test_vocab()
        self.dataset_vocab = self.get_dataset_vocab()
    # def get_train_vocab(self):
    #     all_words = []
    #     for section in self.train_sections:
    #         data = load_textgrid(section=section)
    #         wordlist = extract_word_timestamps(data)
    #         all_words += [w['word'] for w in wordlist]
    #     return list(set(all_words))

    def get_train_vocab(self):
        return self.get_vocab(self.train_sections)

    def get_vocab(self,sections):
        """
        get vocab of sections given a list of sections
        :param sections:
        :return:
        """
        all_words = []
        for section in sections:
            data = load_textgrid(section=section)
            wordlist = extract_word_timestamps(data)
            all_words += [w['word'] for w in wordlist]

        V = list(set(all_words))
        V.remove('#')
        return V
    def get_dataset_vocab(self):
        sections = self.train_sections+self.test_sections
        return self.get_vocab(sections)
    def get_test_vocab(self):
        return self.get_vocab(self.test_sections)


def main():
    vocab = Vocab()
    print('test vocab')
    # train_vocab = get_train_vocab()
    # test_vocab = get_test_vocab()
    # dataset_vocab = get_dataset_vocab()

if __name__ == '__main__':
    main()