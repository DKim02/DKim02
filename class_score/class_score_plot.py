import matplotlib.pyplot as plt

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):  # If 'line' is not a header
                data.append([int(word) for word in line.split(',')])
    return data

if __name__ == '__main__':
    # Step #0) Prepare data
    class_kr = read_data('data/class_score_kr.csv')
    class_en = read_data('data/class_score_en.csv')

    midterm_kr, final_kr = zip(*class_kr)
    total_kr = [40 / 125 * midterm + 60 / 100 * final for (midterm, final) in class_kr]

    midterm_en, final_en = zip(*class_en)
    total_en = [40 / 125 * midterm + 60 / 100 * final for (midterm, final) in class_en]

    # Step #1) Plot midterm/final scores as points
    plt.figure(figsize=(10, 5))
    
    # Scatter plot for Korean class 
    plt.scatter(midterm_kr, final_kr, color='red', s=75, alpha=0.7, label='Korean Class', marker='o')
    # Scatter plot for English class 
    plt.scatter(midterm_en, final_en, color='blue', s=75, alpha=0.7, label='English Class', marker='+')
    
    # Set scatter plot details
    plt.xlim(0, 125)
    plt.ylim(0, 100)
    plt.xlabel('Midterm Score')
    plt.ylabel('Final Score')
    plt.grid(True)
    plt.legend(loc='upper left')

    # Step #2) Plot total scores as a histogram
    plt.figure(figsize=(10, 5))
    plt.hist(total_kr, bins=range(0, 105, 5), color='red', alpha=0.9, label='Korean Class')
    plt.hist(total_en, bins=range(0, 105, 5), color='blue', alpha=0.3, label='English Class')
    plt.xlim(0, 100)
    plt.xlabel('Total score')
    plt.ylabel('The number of students')
    plt.legend(loc='upper left')
    plt.grid(False)
    
    # Display the plots
    plt.show()
