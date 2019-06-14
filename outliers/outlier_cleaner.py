#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)
        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    errors = []

    for i in range(len(predictions)):
        errors.append(abs(predictions[i] - net_worths[i]))
    
    cleaned_data = list(zip(ages, net_worths , errors))
    cleaned_data.sort(key = lambda x : x[2])
    
    
    return cleaned_data [:int(len(cleaned_data) - len(cleaned_data)/10)]
