import matplotlib.pyplot as plt

# Given temperature data
temperatures = [
    8.2, 17.4, 14.1, 7.9, 18.0, 13.5, 9.0, 17.8, 13.0, 8.5,
    16.5, 12.9, 7.7, 17.2, 13.3, 8.4, 16.7, 14.0, 9.5, 18.3,
    13.4, 8.1, 17.9, 14.2, 7.6, 17.0, 12.8, 8.0, 16.8, 13.7,
    7.8, 17.5, 13.6, 8.7, 17.1, 13.8, 9.2, 18.1, 13.9, 8.3,
    16.4, 12.7, 8.9, 18.2, 13.1, 7.8, 16.6, 12.5
]

# Task 1: Classify Temperatures
cold = []  
mild = []  
comfortable = []  

for temp in temperatures:
    if temp < 10:
        cold.append(temp)
    elif 10 <= temp < 15:
        mild.append(temp)
    elif 15 <= temp <= 20:
        comfortable.append(temp)

print("Cold temperatures:", cold)
print("Mild temperatures:", mild)
print("Comfortable temperatures:", comfortable)

# Task 2: Answer questions based on the data
num_cold = len(cold)
num_mild = len(mild)
num_comfortable = len(comfortable)

print(f"Number of cold temperatures: {num_cold}")
print(f"Number of mild temperatures: {num_mild}")
print(f"Number of comfortable temperatures: {num_comfortable}")

# Task 3: Convert to Fahrenheit
temperatures_fahrenheit = []

for temp in temperatures:
    fahrenheit = (temp * 9 / 5) + 32
    temperatures_fahrenheit.append(fahrenheit)

print("Temperatures in Fahrenheit:", temperatures_fahrenheit)

# Task 4: Analyze by Time of Day

night = []  
evening = [] 
day = []  

for i, temp in enumerate(temperatures):
    if i % 3 == 0: 
        night.append(temp)
    elif i % 3 == 1:  
        evening.append(temp)
    elif i % 3 == 2:  
        day.append(temp)

average_day_temp = sum(day) / len(day)
print(f"Average daytime temperature: {average_day_temp:.2f}°C")

# Optional

days = list(range(1, len(day) + 1))

plt.figure(figsize=(10, 5))
plt.plot(days, day, marker='o', label='Daytime Temperature')
plt.title('Daytime Temperature Trends')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.show()
