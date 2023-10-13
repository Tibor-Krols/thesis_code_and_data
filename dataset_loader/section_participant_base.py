from dataset_loader.dataset import LPPDataset
from preprocessing.audio.extract_timestamps_words_audio import extract_sentences

class BaseSectionParticipant:
    def __init__(self,section_participant_dict,include_volume_words_dict=False):
        self.fmri = section_participant_dict['cog_sequence']
        self.participant = section_participant_dict['subject']
        self.labels = section_participant_dict['labels']
        self.section = section_participant_dict['section']
        self.sentences = extract_sentences(self.labels)
        if include_volume_words_dict:
            self.volume_words_dict = self.get_words_volume_idx_dict()
        self.section_vocab = self.get_section_vocab()
        self.nr_fmri_frames =self.fmri.shape[3]
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


    def get_words_volume_idx(self,volume_idx):
        """
        TODO: improve efficiency
        :param volume_idx:
        :return:
        """
        return [l['word'] for l in self.labels if volume_idx in l['volume_idx'] and l['word']!= '#']
    def get_words_volume_idx_dict(self):
        """
        gets dict where keys are indices of the fmri volume and value the words that are said in this volume
        :return:
        """
        return dict([(i,self.get_words_volume_idx(i)) for i in range(self.fmri.shape[3])])

    def get_section_vocab(self):
        return list(set([l['word'] for l in self.labels]))


    def get_volumes_per_word(self,word):
        indices = [l['volume_idx'] for l in self.labels if l['word']==word]
        return [item for sublist in indices for item in sublist]


def main():
    dataset = LPPDataset()
    ps = BaseSectionParticipant(dataset[0],include_volume_words_dict=True)
    vol = ps[0]
    sent_idx = 4
    sent = ps.get_sentence(sent_idx)
    volumes_idx = ps.get_sentence_volumes(sent_idx)
    sent_text = ps.get_sentence_text(sent_idx)


    data = dataset.get_participant_section_data('sub-EN057',1)

    print('end')
    print(ps.fmri.shape)
    avg_sect = torch.mean(ps.fmri,axis = 3)
    print(avg_sect.shape)

if __name__ == '__main__':
    main()