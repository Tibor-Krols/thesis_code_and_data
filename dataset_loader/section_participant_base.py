from dataset_loader.dataset import LPPDataset
from preprocessing.audio.extract_timestamps_words_audio import extract_sentences

class BaseSectionParticipant:
    def __init__(self,section_participant_dict):
        self.fmri = section_participant_dict['cog_sequence']
        self.participant = section_participant_dict['subject']
        self.labels = section_participant_dict['labels']
        self.section = section_participant_dict['section']
        self.sentences = extract_sentences(self.labels)
    def __getitem__(self, index:int):
        """
        get a certain volume of the fmri timeseries based on the index
        :param index:
        :return:
        """
        return self.fmri[..., index]


    def get_sentence(self,index):
        return self.sentences[index]

    def get_sentence_volumes(self,index):
        """
        gets list of volume indexes for a sentence with a given index for a certain segment
        :param index:
        :return:
        """
        sent = self.sentences[index]
        volumes_list = [w['volume_idx'] for w in sent]
        volumes = list(set([item for sublist in volumes_list for item in sublist]))
        return volumes

    def get_sentence_text(self,index):
        sent = self.sentences[index]
        return " ".join([w['word'] for w in sent])

    def get_nr_sentences(self):
        return len(self.sentences)



def main():
    dataset = LPPDataset()
    ps = BaseSectionParticipant(dataset[0])
    vol = ps[0]
    sent_idx = 4
    sent = ps.get_sentence(sent_idx)
    volumes_idx = ps.get_sentence_volumes(sent_idx)
    sent_text = ps.get_sentence_text(sent_idx)

    data = dataset.get_participant_section_data('sub-EN057',1)

    print('end')

if __name__ == '__main__':
    main()