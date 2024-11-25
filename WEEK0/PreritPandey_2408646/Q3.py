# Task 1: Coin Change Problem 
def min_coins_recursive(coins, amount):
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')

    min_coins = float('inf')
    for coin in coins:
        result = min_coins_recursive(coins, amount - coin)
        if result != float('inf'):
            min_coins = min(min_coins, result + 1)
    
    return min_coins if min_coins != float('inf') else -1

# Task 2
def lcs_recursive(s1, s2, i, j):
    if i == 0 or j == 0:
        return 0
    if s1[i - 1] == s2[j - 1]:
        return 1 + lcs_recursive(s1, s2, i - 1, j - 1)
    else:
        return max(lcs_recursive(s1, s2, i - 1, j), lcs_recursive(s1, s2, i, j - 1))

def longest_common_subsequence_recursive(s1, s2):
    return lcs_recursive(s1, s2, len(s1), len(s2))

# Task 3
def knapsack_recursive(weights, values, capacity, n):
    if n == 0 or capacity == 0:
        return 0
    if weights[n - 1] > capacity:
        return knapsack_recursive(weights, values, capacity, n - 1)
    else:
        include = values[n - 1] + knapsack_recursive(weights, values, capacity - weights[n - 1], n - 1)
        exclude = knapsack_recursive(weights, values, capacity, n - 1)
        return max(include, exclude)

def knapsack_recursive_solution(weights, values, capacity):
    return knapsack_recursive(weights, values, capacity, len(weights))


# Task 1
coins = [1, 2, 5]
amount = 11
print("Task 1 - Minimum coins needed (Recursive):", min_coins_recursive(coins, amount))  

# Task 2
s1 = "abcde"
s2 = "ace"
print("Task 2 - Length of LCS (Recursive):", longest_common_subsequence_recursive(s1, s2))  

# Task 3
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7
print("Task 3 - Maximum value in knapsack (Recursive):", knapsack_recursive_solution(weights, values, capacity))  
