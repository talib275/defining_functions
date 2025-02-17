import numpy as np
import string
import pandas as pd
import re

########################################################

# All functions for Q1
def all_rounder(seq, method, arg=None):  
    """
    Executes various operations on sequences (lists, tuples, strings, dictionaries) that are inputted.

    The operations suitable are:
        List methods: append, extend, insert, remove, pop, clear, reverse, sort
        String methods: join, replace, split, find, upper, lower, strip
        Tuple methods: count, index
        Dictionary methods: get, pop, update, clear, keys, values, items

    Parameters:
        seq (list, tuple, str, dict): the input to be formatted.
        method (function): used to format the sequence (supports different sequencee types and methods).
        arg (optional): the second input for formatting (default is none).
            takes any data type like "seq" does.
            this input isn't required in all situations (when the method is only works for one sequence e.g. '.lower()').

    Returns:
        The formatted sequence .
    """       
    method = method.lower()                    
    
    # Convert the string to a tuple or list (in case the input was read as string but isn't)
    if isinstance(arg, str) and (arg.startswith("(") or arg.startswith("[")):
        arg = eval(arg)  

   
    if isinstance(seq, str) and (seq.startswith("(") or seq.startswith("[")):          
        seq = eval(seq)  

    # Specific programs for "insert" and "replace" methods with tuples to ensure they're done properly
    if method == "insert" and isinstance(arg, tuple) and len(arg) == 2:                 
        code_str = f"seq.{method}({arg[0]}, {repr(arg[1])})"                           

    elif method == "replace" and isinstance(arg, tuple) and len(arg) == 2:              
        code_str = f"seq.{method}({repr(arg[0])}, {repr(arg[1])})"                      

    # Determine whether "arg" input is required
    elif arg is None:                                                                   
        code_str = f"seq.{method}()"                                                    
    else:
        code_str = f"seq.{method}({repr(arg)})"                                       

    # Always use eval() for tuples since they return values rather than modifying in place
    if isinstance(seq, tuple):
        result = eval(code_str)
        return result

    # eval() is used because these methods produce a value that needs to be captured and returned
    elif method in ["upper", "lower", "strip", "replace", "join", "split", "find"]:     
        result = eval(code_str)
        return result
    # exec() is used for methods that don't return a value; they modify the sequence in place
    else:
        exec(code_str)                                                                 
        return seq
    

# All functions for Q2
def padded_broadcasting(func, a, b, pad=1):
    """
    Handles array broadcasting when standard broadcasting fails.
    Compares the dimensions of two arrays and will. 
    Pads the smaller array with 1 (by default) or the user can input their own value.
    Computes the given NumPy function on the two arrays.

    Parameters:
        func (function): The NumPy function to apply (e.g., np.add).
        a (array-like): First array input.
        b (array-like): Second array input.
        pad (int, optional): Value used for padding. Default is 1.

    Returns:
        ndarray: The result of applying `func` to the arrays with necessary padding (multi-dimensional array).
        None: If standard broadcasting applies.
    """
    
    # Ensures inputs are read as arrays
    if not hasattr(a, 'shape'):
        a = np.array(a)
    if not hasattr(b, 'shape'):
        b = np.array(b)

    # Check if arrays are broadcastable
    shape1 = (1,) * (len(b.shape) - len(a.shape)) + a.shape
    shape2 = (1,) * (len(a.shape) - len(b.shape)) + b.shape
    broadcastable = all(dim1 == dim2 or dim1 == 1 or dim2 == 1 for dim1, dim2 in zip(shape1, shape2))

    if broadcastable:
        print("Standard broadcasting should be applied!")
        return None

    # Adjust dimensions to match shapes
    dimension = max(len(a.shape), len(b.shape))
    new_a = (1,) * (dimension - len(a.shape)) + a.shape
    new_b = (1,) * (dimension - len(b.shape)) + b.shape

    # Determine the maximum shape for padding
    max_shape = np.maximum(new_a, new_b)

    # Pad arrays
    pad_width_a = [(0, target - current) for current, target in zip(a.shape, max_shape)]
    pad_width_b = [(0, target - current) for current, target in zip(b.shape, max_shape)]
    padded_a = np.pad(a, pad_width=pad_width_a, mode='constant', constant_values=pad)
    padded_b = np.pad(b, pad_width=pad_width_b, mode='constant', constant_values=pad)

    # Apply the function
    result = func(padded_a, padded_b)
    return result

# All functions for Q3
import numpy as np

def txtanalyser(fname, t, f, sel):
    """
    Analyses the text using Numpy functions to find the count of the chosen word, and the statistical data of the minutes.

    Parameters:
        fname (.txt): file to analyse.
        t (str): chosen word for the calculations.
        f (function): NumPy function for calculation on the minutes (e.g. np.mean, np.max etc.).
        sel (str): count/find for the file:
            "count" determines the number of times the given word ("t") is written; f is ignored when sel is "count".
            "find" is used for the np. functions.

    Returns:
        when sel is "count": the total numnber of times a given word is found in the text.
        when sel is "find": the np. function is used to calculate an output for the minutes (e.g. np.mean will give the mean minute that the word was found).
    """

    # load file
    txt = np.genfromtxt(fname, dtype=[("Minute", float), ("Commentary", "U200")], delimiter="\t", encoding="utf-8", skip_header=1)

    # split the two columns of data
    mins = txt["Minute"]
    cmnt = txt["Commentary"]

    # only give the sum of words on the line where input "t" is found
    if sel == "count":
        return float(np.sum([1 for line in cmnt if f" {t} " in f" {line} "]))
    
    # calculates the minutes where the word is found
    elif sel == "find":
        filtered_mins = [mins[i] for i, line in enumerate(cmnt) if f" {t} " in f" {line} "]
        if filtered_mins:
            return float(f(filtered_mins))
        else:
            return None
    else:
            raise ValueError("'sel' must be 'count' or 'find'")
    
# All functions for Q4
def find_alphabetical_order(file_path, check_ties=True):
    """
    Finds the alphabetical order of a line by assessing the first character of each word (if check_ties is false).
    If check_ties is True: each letter is assessed (if the first letters are the same).

    Parameters:
        file_path (.txt): file to analyse.
        check_ties (bool): when true, each letter of each word is assessed to determine correct alphabetical order.

    Returns:
        string in alphabetical order.
            letter by letter when check_ties is true.
    """

    # retrieve file
    text_file = np.genfromtxt(file_path, dtype=[("Minute", float), ("Commentary", "U200")], delimiter="\t", encoding="utf-8", skip_header=1)

    # retrieve the "Commentary" column
    text = text_file["Commentary"]

    # clean text
    cleaned_text = []
    accent_map = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ñ': 'n', 'ü': 'u', 'Á': 'A', 'É': 'E', 'Í': 'I',
        'Ó': 'O', 'Ú': 'U', 'Ñ': 'N', 'Ü': 'U'
    }

    for line in text:
        # remove punctuation
        line = re.sub(r"[^\w\s]", "", line)  
        
        # replace accented characters 
        line = ''.join(accent_map.get(char, char) for char in line)

        cleaned_text.append(line)

    # create empty list to store the computed string
    alphabetical_lines = []

    for line in cleaned_text:
        words = line.split()
        is_alphabetical = True

        for i in range(len(words) - 1):

            word1, word2 = words[i], words[i + 1]

            # alphabetically arrange the words by assessing the letters in each word (not just the first letter) when check_ties is true
            if check_ties:
                for char1, char2 in zip(word1, word2):
                    if char1.lower() != char2.lower():
                        if char1.lower() > char2.lower():
                            is_alphabetical = False
                        break
            else:
                if word1[0].lower() > word2[0].lower():
                    is_alphabetical = False

            if not is_alphabetical:
                break

        # add the alphabetical text into a new list
        if is_alphabetical:
            alphabetical_lines.append(line)

    return alphabetical_lines

# All functions for Q5
def pd_query():
    """
    Provides specific data within "olympics.csv" that were asked for.

    The queries are:
        q1 (df): Descriptive statistics for BMI and age_competing of countries in descending order of mean age_competing values.
        q2 (str): The athlete name in str who wins the highest number of medals.
        q3 (df): 61 tallest bronze medallists (descending by age_competing).
        q4 (df): The medallists of ski jumping in the year where the greatest number of countries participated (in the sport).
        q5 (df): The dataset for the olympian with the greatest range in age_competing (i.e. participated in the olympics for the longest time).

    Parameters:
        User inputs any query (or section of query) to retrieve and analyse that particular data.

    Returns:
        Predefined queries based on the query inputted (asked for).
    """
    # Load the dataset
    df = pd.read_csv('olympics.csv')

    # query 1
    q1 = df.groupby('country_noc')[['BMI', 'age_competing']].describe().sort_values(('age_competing', 'mean'), ascending=False).dropna(subset=[('BMI', 'mean')])

    # query 2
    q2 = df[df['medal'].notna()]['name'].value_counts().idxmax()

    # query 3
    q3 = df[(df['medal'] == 'Bronze') & (df['BMI'] > 50)].nlargest(61,'height').sort_values('age_competing', ascending=False).drop(columns=['Unnamed: 0'])

    # query 4
    q4 = df[df['sport'] == 'Ski Jumping'][df[df['sport'] == 'Ski Jumping']['year'] == df[df['sport'] == 'Ski Jumping'].loc[df[df['sport'] == 'Ski Jumping']['Num_countries'].idxmax(), 'year']][['name', 'medal', 'year', 'event_title']].dropna(subset=[('medal')])


    # query 5
    q5=df[df['athlete_id'] == (df.groupby('athlete_id')['age_competing'].agg(['min', 'max'])['max'] - df.groupby('athlete_id')['age_competing'].agg(['min', 'max'])['min']).idxmax()].drop(columns=['Unnamed: 0'])



    return q1, q2, q3, q4, q5