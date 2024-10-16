questionnaires = {
    'mmse': {
        'name': 'MMSE',
        'criteria': [
            'MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY', 'MMSEASON',
            'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE',
            'WORD1', 'WORD2', 'WORD3', 'MMD', 'MML', 'MMR', 'MMO', 'MMW',
            'WORD1DL', 'WORD2DL', 'WORD3DL',
            'MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR',
            'MMREAD', 'MMWRITE', 'MMDRAW'
        ],
        'criteria_range': [[0, 1] for _ in range(30)],
        'questions': [
            'What is the date today?', 'What year is it?', 'What month is it?', 'What day of the week is it?', 'What season is it?',
            'What hospital are we in?', 'What floor are we on?', 'What city are we in?', 'What area are we in?', 'What state are we in?',
            'Word 1 recall', 'Word 2 recall', 'Word 3 recall', 'D', 'L', 'R', 'O', 'W',
            'Word 1 delayed recall', 'Word 2 delayed recall', 'Word 3 delayed recall',
            'Identify a watch', 'Identify a pencil', 'Repeat "No ifs, ands, or buts"',
            'Follow a three-step command (hand)', 'Follow a three-step command (fold)', 'Follow a three-step command (on floor)',
            'Read and obey "Close your eyes"', 'Write a sentence', 'Copy a design'
        ]
    },
    'moca': {
        'name': 'MOCA',
        'criteria': [
            'TRAILS','CUBE',
            'CLOCKCON','CLOCKNO','CLOCKHAN',
            'LION','RHINO','CAMEL',
            'IMMT1W1','IMMT1W2','IMMT1W3','IMMT1W4','IMMT1W5','IMMT2W1','IMMT2W2','IMMT2W3','IMMT2W4','IMMT2W5',
            'DIGFOR','DIGBACK',
            'LETTERS',
            'SERIAL1','SERIAL2','SERIAL3','SERIAL4','SERIAL5',
            'REPEAT1','REPEAT2',
            'FFLUENCY',
            'ABSTRAN','ABSMEAS',
            'DELW1','DELW2','DELW3','DELW4','DELW5',
            'DATE','MONTH','YEAR','DAY','PLACE','CITY'
        ],
        'criteria_range': [[0, 1] for _ in range(37)],  # Adjusted length to match the number of criteria
        'questions': [
            'Draw a line to connect alternating letters and numbers (e.g., 1-A, 2-B, etc.)', 'Copy the cube',
            'Draw a clock with the contour, numbers, and hands correctly placed',
            'Place the numbers correctly on the clock face', 'Set the clock hands to 10 past 11',
            'Name the animal shown below (Lion)', 'Name the animal shown below (Rhinoceros)', 'Name the animal shown below (Camel)',
            "Immediate recall of 'face'", "Immediate recall of 'velvet'", "Immediate recall of 'church'",
            "Immediate recall of 'daisy'", "Immediate recall of 'red'",
            "Delayed recall of 'face'", "Delayed recall of 'velvet'", "Delayed recall of 'church'",
            "Delayed recall of 'daisy'", "Delayed recall of 'red'",
            "Repeat the numbers in the same forward order (2 1 8 5 4)", "Repeat the numbers in reverse order (7 4 2)",
            'Tap for each letter "A" in the sequence',
            'Subtract 7 from 100 (Answer: 93)', 'Subtract 7 from 93 (Answer: 86)', 'Subtract 7 from 86 (Answer: 79)',
            'Subtract 7 from 79 (Answer: 72)', 'Subtract 7 from 72 (Answer: 65)',
            'Repeat: "I only know that John is the one to help today."',
            'Name more than 11 words that begin with the letter "F"',
            'Find the similarity between train and bicycle', 'Find the similarity between watch and ruler',
            'What is the date today?', 'What is the month?', 'What is the year?', 'What day of the week is it?', 'Where are we right now (place)?', 'What city are we in?'
        ]
    },
    'npiq': {
        'name': 'NPIQ',
        'criteria': [
            'NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV',
            'NPIFSEV', 'NPIGSEV', 'NPIHSEV', 'NPIISEV', 'NPIJSEV',
            'NPIKSEV', 'NPILSEV'
        ],
        'criteria_range': [[0, 1, 2, 3] for _ in range(12)],
        'questions': [
            'Delusions: Does the patient have false beliefs, such as thinking that others are stealing from him/her or planning to harm him/her in some way?',
            'Hallucinations: Does the patient have hallucinations such as false visions or voices? Does he or she seem to hear or see things that are not present?',
            'Agitation/Aggression: Is the patient resistive to help from others at times, or hard to handle?',
            'Depression/Dysphoria: Does the patient seem sad or say that he/she is depressed?',
            'Anxiety: Does the patient become upset when separated from you? Does he/she have any other signs of nervousness such as shortness of breath, sighing, being unable to relax, or feeling excessively tense?',
            'Elation/Euphoria: Does the patient appear to feel too good or act excessively happy?',
            'Apathy/Indifference: Does the patient seem less interested in his/her usual activities or in the activities and plans of others?',
            'Disinhibition: Does the patient seem to act impulsively, for example, talking to strangers as if he/she knows them, or saying things that may hurt people\'s feelings?',
            'Irritability/Lability: Is the patient impatient and cranky? Does he/she have difficulty coping with delays or waiting for planned activities?',
            'Motor Disturbances: Does the patient engage in repetitive activities such as pacing around the house, handling buttons, wrapping string, or doing other things repeatedly?',
            'Night-time Behaviors: Does the patient awaken you during the night, rise too early in the morning, or take excessive naps during the day?',
            'Appetite/Eating: Has the patient lost or gained weight, or had a change in the type of food he/she likes?'
        ]
    }
}

if __name__ == '__main__':
    print(len(questionnaires['moca']['criteria']))