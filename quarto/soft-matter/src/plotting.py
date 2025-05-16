import matplotlib.pyplot as plt
import inspect

# def brackify(string):


def simple_plot(arr1, arr2):
    # Get caller local variables for name lookup
    fig = plt.figure(figsize=(5, 2))
    caller_locals = inspect.currentframe().f_back.f_locals
    # Find names by matching objects
    arr1_name = next((name for name, val in caller_locals.items() if val is arr1), 'arr1')
    arr2_name = next((name for name, val in caller_locals.items() if val is arr2), 'arr2')
    plt.plot(arr1,arr2)
    
    if "_"  in arr1_name:   
        arr1_name = arr1_name.replace("_", r"_{")
        arr1_name+=r"}"

    if "_"  in arr2_name:   
        arr2_name = arr2_name.replace("_", r"_{")
        arr2_name+=r"}"
        
    plt.xlabel(rf"${arr1_name}$")
    plt.ylabel(rf"${arr2_name}$")
    plt.tight_layout()
    plt.show()