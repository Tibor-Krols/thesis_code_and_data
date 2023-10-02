from preprocessing.audio.extract_timestamps_words_audio import load_audio_timestamps, extract_sentences,load_all_sections_timestamps


# section_dict = {}
# for section in range(1,10):
#     print(section)
#     wordslist = load_audio_timestamps(section=section)
#     section_dict[section] = wordslist


#TODO: make sentence dict where each sent has
# TODO: keep it per section, because fmri is per section and timestamps are relative to start fmri section
# TODO: move to extract_sentences?
# loop over sections:
#     loop over sentences:
# {
#     'sentence_id': unique_id f"{section}_{sentence_no}",
#     'section': which audio section its from
#     'words': sent_wordlist with start and end time,
#     'volumes': [volume_ids ] which volumes it entails,
#     'start': sent_wordlist[0]['start'],
#     'end': sent_wordlist[-1]['end']
# }

def main():
    sections_timestamps_dict = load_all_sections_timestamps()
    section = 1
    sentences_sec1 = extract_sentences(wordlist=sections_timestamps_dict[section])
    print('test')

if __name__ == '__main__':
    main()
