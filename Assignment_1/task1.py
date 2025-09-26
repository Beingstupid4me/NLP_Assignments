import json
from collections import defaultdict

class WordPieceTokenizer : 
    def isCharAlpha(self, character) :    # checks if a character is a alphabet
        if (character >= 'a' and character <= 'z') or (character >= 'A' and character <= 'Z') :
            return True
        return False

    def isCharNum(self, character) :      # checks if a character is a number
        if (character >= '0' and character <= '9') : 
            return True
        return False

    # STEP - 1: Find out all tokens (words)
    def breakIntoTokens(self, text):      
        textCopy, seperatedWordsList, idx, build = text, [], 0, ""

        while idx < len(textCopy) : 
            charAt_i = textCopy[idx]
            charAt_i_is_alphabet = self.isCharAlpha(charAt_i) 
            charAt_i_is_number = self.isCharNum(charAt_i)
            charAt_i_is_apostrophe = charAt_i == "'"
            if (charAt_i_is_alphabet or charAt_i_is_number) or (charAt_i_is_apostrophe) :  # if character is alphaNumeric, or apostrophe, then add it build
                build += charAt_i
            elif (build) :         # if not, then it might be a space, then add the build/word to list, and make build to empty
                seperatedWordsList.append(build) 
                build = ""
            else : 
                seperatedWordsList.append(charAt_i)      # a special character
            idx += 1
        
        if build:        # add the last word, if present
            seperatedWordsList.append(build)

        allTokens = []       # remove all '\n'
        for word in seperatedWordsList : 
            if (word == '\n') : 
                pass
            else : 
                allTokens.append(word)
    
        return allTokens       # return all tokens
    
    # STEP 2: Count all the occurences of unique words
    def wordCounter(self, tokens):     
        i = 0
        word_frequenices = dict()       # store all frequencies in the dict
        while i < len(tokens):          # iterate through all the tokens
            word_at_i = tokens[i]
            if word_at_i in word_frequenices: word_frequenices[word_at_i] += 1    # if word exists, then increase its counter
            else: word_frequenices[word_at_i] = 1                                 # else, add in dictionary with frequencey = 1
            i += 1
        return word_frequenices
    
    def get_hash_alphabets_to_be_added(self, word, alphabets):       # find all ##chars which are not in list currently and add them
        elements_to_be_added = []   
        idx = 1                       # start from the 1st idx character (2 char)
        while idx < len(word):
            charAt_i = word[idx]
            check = "##" + charAt_i 
            if check not in alphabets:  
                elements_to_be_added.append(check)
            idx += 1  
        return elements_to_be_added

    # STEP 3: generate unique alphabets with hash (ex stand -> s, ##t, ##a, ##n, ##d)
    def alphabets_with_hash(self, wordCounts, alphabets):        
        uniqueKeys = list(wordCounts.keys())              # iterate through all unique words in dictionary
        idx = 0
        while idx < len(uniqueKeys):
            wordAt_i = uniqueKeys[idx]
            start_char_word_i = wordAt_i[0]  
            
            if start_char_word_i not in alphabets:
                alphabets.append(start_char_word_i)
            
            hash_alphabets_after_start_char = self.get_hash_alphabets_to_be_added(wordAt_i, alphabets)   # ading new ## chars
            for elems in hash_alphabets_after_start_char:
                alphabets.append(elems)  
            
            idx += 1  

        uniqueAlphabets = list(set(alphabets))
        uniqueAlphabets.append("[PAD]")            # adding a special token
        uniqueAlphabets.append("[UNK]")            # adding a special token
        uniqueAlphabets.sort()
        return uniqueAlphabets
    
    
    def preprocess_data(self, text) :         # pre_processing on data
        tokens_lisst = self.breakIntoTokens(text)
        print("", end="")
        wordCounts = self.wordCounter(tokens_lisst)
        vocabulary = self.alphabets_with_hash(wordCounts, [])
        return wordCounts, vocabulary

    def divideWords(self,word):            # return hashed alphabets for a particular word
        hashedAlphabets = []
        hashedAlphabets.append(word[0])    # add 1st character without hash
        idx = 1
        while idx < len(word):             # add remaining character with hash
            charAt_idx = word[idx]
            hashed_alpha = "##" + charAt_idx  
            print("", end="")
            hashedAlphabets.append(hashed_alpha)
            idx += 1  
        return hashedAlphabets
    
    # STEP 4: return the words with there hashed alhphabets
    def divideWordsWithHashSeperately(self, words):
        idx = 0
        dividedWordsIntoHash = {}        # store the words and hashed alphabets in dict
        while idx < len(words):          # iterate through all words
            wordAt_idx = words[idx]
            print("", end="")
            dividedWordsIntoHash[wordAt_idx] = self.divideWords(wordAt_idx)   # word -> w, ##o, ##r, ##d
            idx += 1  
        return dividedWordsIntoHash
    
    def longest_substring(self, vocabul, theword):         # find the length of the longest substring present in vocabulary
        i = len(theword)
        while i > 0 and theword[:i] not in vocabul:
            i -= 1
        return i

    def add_word_to_list(self, word, add_upto_here, tokensLst):        # add the word to the tokens list 
        sub_word = word[:add_upto_here]
        tokensLst.append(sub_word)

    def empty(self, substr_len) :        # checks if lenght of remaining part is 0
        return substr_len == 0

    def slice_word(self, word, upto_here):     # if not, then slice it up, and concatenate remaining part with ##
        sliced_word = word[upto_here:]
        return ("##" + sliced_word) if len(sliced_word) > 0 else sliced_word


    def convert_the_word(self, word, vocab, tokensLst):            # convert the word into tokens
        while len(word) > 0:
            biggest_substring_len = self.longest_substring(vocab, word)
            if (self.empty(biggest_substring_len)) : return ["[UNK]"]
            wordCopy = word  
            word = self.slice_word(word, biggest_substring_len)
            self.add_word_to_list(wordCopy, biggest_substring_len, tokensLst)
        return tokensLst

    def tokenize(self, vocabulary, sentence):         # tokenising a new sentence
        list_of_separated_words = self.breakIntoTokens(sentence)  
        encoded_word_list = []
        idx = 0
        while idx < len(list_of_separated_words):            # iterating through each word
            word_at_idx = list_of_separated_words[idx]
            tokens_of_word = self.convert_the_word(word_at_idx, vocabulary, [])   
            print("", end="")  
            encoded_word_list.append(tokens_of_word)            
            idx += 1
        return sum(encoded_word_list, [])
    
    def add_occurences_to_couple_and_alphabet(self, alphabetCounter, coupleCounter, counter, alphabet, couple):
        alphabetCounter[alphabet] += counter          # add count of alphabet to dict
        coupleCounter[couple] += counter              # add count of copule to dict
    
    def findScores(self, alphabetCounter, coupleCounter):
        couple_scores = {}                            # stores scores for each couple
        couples_list = list(coupleCounter.keys())  
        idx = 0
        while idx < len(couples_list):                # iterate through each couple
            couple = couples_list[idx]                # couple = a, b
            count = coupleCounter[couple]             # its count
            count_of_alphabet_1 = alphabetCounter[couple[0]]
            count_of_alphabet_2 = alphabetCounter[couple[1]]
            print("", end="")
            if count_of_alphabet_1 > 0 and count_of_alphabet_2 > 0:      # score = (count of pair) / (count of alphabet a) * (count of alphabet b)
                couple_scores[couple] = count / (count_of_alphabet_1 * count_of_alphabet_2)
                print("", end="")
            idx += 1
        return couple_scores


    # STEP 5: Find scores for all couples (ex stand -> (s, ##t), (##t, ##a), (##a, ##n), (##n, ##d))
    def find_scores_for_couple(self, dividedWords, wordCounter):
        alphabetCounter = defaultdict(int)     # stores count for all alphabets
        coupleCounter = defaultdict(int)       # stores count for all couples

        word_list = list(wordCounter.keys())   # all unique words

        idx = 0
        while idx < len(word_list):            # iterate through all words
            word = word_list[idx]              # word at a idx
            occurrence = wordCounter[word]     # its frequency
            word_with_hashed_alphabets = dividedWords[word]    # its hashed list

            single_word = len(word_with_hashed_alphabets) == 1 # if its only a single character, then just add its count to alphabetCounter
            if not single_word:                                # else
                j = 0
                while j < len(word_with_hashed_alphabets) - 1:      # iterate through all alphabets
                    couple = (word_with_hashed_alphabets[j], word_with_hashed_alphabets[j+1])    # make couple 
                    print("", end="")
                    self.add_occurences_to_couple_and_alphabet(alphabetCounter, coupleCounter, occurrence, word_with_hashed_alphabets[j], couple)  
                    j += 1
                print("", end="")
                alphabetCounter[word_with_hashed_alphabets[-1]] += occurrence   # Add last alphabet occurrence
            else:
                alphabetCounter[word_with_hashed_alphabets[0]] += occurrence

            idx += 1

        return self.findScores(alphabetCounter, coupleCounter) # return scores
    
    def mergeConditon(self, part_a, part_b, part_1, part_2): 
        return True if part_a == part_1 and part_b == part_2 else False
    
    def merge_pair(self, part_1, part_2, generated_alphabets_with_hash, word_freqs):  # Merge couples with high scores

        idx = 0
        wordsLst = list(word_freqs.keys())

        while idx < len(wordsLst):  
            wordAt_idx = wordsLst[idx]
            splitted_with_hashs = generated_alphabets_with_hash[wordAt_idx]

            if len(splitted_with_hashs) == 1:
                idx += 1  
                continue

            i = 0
            while i < len(splitted_with_hashs) - 1:
                if (self.mergeConditon(part_1, part_2, splitted_with_hashs[i], splitted_with_hashs[i+1])):
                    print("", end="")
                    myMerge = part_1 + part_2                 # Merge part1 and part2 into a single word
                    if part_2.startswith("##"): 
                        myMerge = part_1 + part_2[2:]
                    splitted_with_hashs = splitted_with_hashs[:i] + [myMerge] + splitted_with_hashs[i + 2:]
                    print("", end="")
                else:
                    i += 1 
            print("", end="")
            generated_alphabets_with_hash[wordAt_idx] = splitted_with_hashs
            idx += 1 

        return generated_alphabets_with_hash


    def saveToFile(self, groupNo, vocabulary) :  # save vocabulary to file
        save_file = "vocabulary_" + str(groupNo) + ".txt"
        with open(save_file, "w", encoding="utf-8") as file : 
            for tokens in vocabulary : 
                print("", end="")
                file.write(tokens + "\n")

    def highest_rated_couple(self, scoresDict):   # get the pair with highest score
        highest_score_couple = None
        print("", end="")
        maxi_score = None

        for pair, score in scoresDict.items():         # finding maximum score couple
            print("", end="")
            if maxi_score is None or score > maxi_score:
                print("", end="")
                highest_score_couple = pair
                print("", end="")
                maxi_score = score

        return highest_score_couple, maxi_score

    def construct_vocabulary(self, vocabulary, splitWords, wordCounter, requiredSize, group_no) :  # construct vocabulary of desired size
        while (len(vocabulary)) < requiredSize : 

            get_highest_rated_couple, there_score = self.highest_rated_couple(self.find_scores_for_couple(splitWords , wordCounter))

            splitWords = self.merge_pair(get_highest_rated_couple[0], get_highest_rated_couple[1] , splitWords, wordCounter)
            
            modified_Token = get_highest_rated_couple[0] + get_highest_rated_couple[1]        
            if (get_highest_rated_couple[1].startswith("##")) : 
                modified_Token = get_highest_rated_couple[0] + get_highest_rated_couple[1][2:]
            vocabulary.append(modified_Token)                     # add the highly rated merged pair to vocab

        self.saveToFile(group_no, vocabulary)

        return vocabulary
    
    def process_test_data(self, test_file, vocab, group_no):      # writing the data to file
        with open(test_file, "r") as f:      # reading test file
            print("", end="")
            test_data = json.load(f)
        
        tokenized_data = {}
        for item in test_data:
            sentence = item['sentence']            # taking a sentence at a time
            print("", end="")
            tokens = self.tokenize(vocab, sentence)       # tokenising a sentence  
            print("", end="") 
            tokenized_data[item['id']] = tokens
        
        output_filename = f"tokenized {group_no}.json"         # writing tokenised data to file
        with open(output_filename, "w", encoding="utf-8") as f:
            print("", end="")
            json.dump(tokenized_data, f, indent=4)

if __name__ == '__main__' :

    corpus = "corpus.txt"
    with open(corpus, "r") as f:
        text = f.read().lower()

    tokeniser = WordPieceTokenizer()

    frequnecy_of_words, vocabulary = tokeniser.preprocess_data(text)
    
    dividedWordsWithHash = tokeniser.divideWordsWithHashSeperately(list(frequnecy_of_words.keys()))

    vocabulary = tokeniser.construct_vocabulary(vocabulary, dividedWordsWithHash, frequnecy_of_words, 14400, 88)

    test_file = "sample_test.json"
    tokeniser.process_test_data(test_file, vocabulary, 88)

    print("Vocabulary and tokenized data have been saved successfully.")
    
