temperatures = [
    8.2, 17.4, 14.1, 7.9, 18.0, 13.5, 9.0, 17.8, 13.0, 8.5,
    16.5, 12.9, 7.7, 17.2, 13.3, 8.4, 16.7, 14.0, 9.5, 18.3,
    13.4, 8.1, 17.9, 14.2, 7.6, 17.0, 12.8, 8.0, 16.8, 13.7,
    7.8, 17.5, 13.6, 8.7, 17.1, 13.8, 9.2, 18.1, 13.9, 8.3,
    16.4, 12.7, 8.9, 18.2, 13.1, 7.8, 16.6, 12.5
]

night_temperatures = []
evening_temperatures = []
day_temperatures = []

for i in range(len(temperatures)):
    if i % 3 == 0:
        night_temperatures.append(temperatures[i])
    elif i % 3 == 1:
        evening_temperatures.append(temperatures[i])
    elif i % 3 == 2:
        day_temperatures.append(temperatures[i])

average_day_temperature = sum(day_temperatures) / len(day_temperatures)

print("Night temperatures:", night_temperatures)
print("Evening temperatures:", evening_temperatures)
print("Day temperatures:", day_temperatures)
print(f"Average day-time temperature: {average_day_temperature:.2f}°C")