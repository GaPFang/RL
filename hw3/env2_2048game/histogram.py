import matplotlib.pyplot as plt

# Data provided
data = {
    32: 1,
    64: 8,
    128: 47,
    256: 41,
    512: 3
}

# Prepare the data for plotting
x_labels = list(data.keys())   # Categories as labels
y = list(data.values())        # Frequencies

# Plotting with equal spacing for categories
plt.bar(range(len(x_labels)), y, color='blue')
plt.xlabel('Best tile')
plt.ylabel('Frequency')
plt.title('Best tile distribution')

# Set custom tick labels to have equal spacing
plt.xticks(range(len(x_labels)), x_labels)

# Save and show the figure
plt.savefig("histogram.png", format='png')
