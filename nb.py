"""
Naive Bayes classifier

- gets reviews as input
- counts how many times words appear in pos/neg
- adds one to each (to not have 0 probabilities)
- computes likelihood and multiply by prior (of review being pos/neg) to get the posterior probability
- in a balanced dataset, prior is the same for both, so we ignore it here
- chooses highest probability to be prediction
"""

import math


def test(words, probs, priors, file_name):
    label = 1 if "pos" in file_name else -1
    count = 0
    correct = 0
    with open(file_name) as file:
        for line in file:
            # begin with prior (simply how likely it is to be pos/neg before evidence)
            pos = priors[0]
            neg = priors[1]
            # compute likelihood
            # sum logs, better than multiplying very small numbers
            for w in line.strip().split():
                # if word wasn't in train data, then we have to ignore it
                # same effect if we add test words into corpus and gave small probability
                if w in words:
                    pos += math.log(probs[w][0])
                    neg += math.log(probs[w][1])
            
            # say it's positive if pos >= neg
            pred = 1 if pos >= neg else -1
            
            # increment counters
            count += 1
            if pred == label:
                correct += 1
    
    # return results
    return 100*(correct/float(count))


def main():
    # count number of occurances of each word in pos/neg reviews
    # we'll use a dict containing a two item list [pos count, neg count]
    words = {}
    w_count = 0 # words
    p_count = 0 # positive instances
    n_count = 0 # negative instances
    
    # count positive occurrences
    with open("data/train.positive") as file:
        for line in file:
            for word in line.strip().split():
                try:
                    words[word][0] += 1
                except:
                    words[word] = [1, 0]
                w_count += 1
            p_count += 1
    
    # count negative occurrences
    with open("data/train.negative") as file:
        for line in file:
            for word in line.strip().split():
                try:
                    words[word][1] += 1
                except:
                    words[word] = [0, 1]
                w_count += 1
            n_count += 1
    
    # calculate probabilities of each word
    corpus = len(words)
    probs = {}
    for key, value in words.iteritems():
        # smooth values (add one to each)
        value[0]+=1
        value[1]+=1
        # prob = count / total count + number of words (for smoothing)
        p_pos = value[0] / float(w_count + corpus)
        p_neg = value[1] / float(w_count + corpus)
        probs[key] = [p_pos, p_neg]
    
    # compute priors based on frequency of reviews
    priors = []
    priors.append(math.log(p_count / float(p_count + n_count)))
    priors.append(math.log(n_count / float(p_count + n_count)))
    
    # test naive bayes
    pos_result = test(words, probs, priors, "data/test.positive")
    neg_result = test(words, probs, priors, "data/test.negative")
    
    print "Accuracy(%)"
    print "Positive:", pos_result
    print "Negative:", neg_result
    print "Combined:", (pos_result+neg_result)/float(2)


if __name__ == "__main__":
    print "-- Naive Bayes classifier --\n"
    
    main()
    