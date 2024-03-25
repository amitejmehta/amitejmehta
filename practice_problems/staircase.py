"""
Problem Statement: You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps.

In how many distinct ways can you climb to the top?

Constraints: n>= k

Framework for Solving DP Problems:
1. Define the objective function and identity its type (combinatorial vs optimization).
    f(i) is the number of distinct ways we can reach step i
2. Identify any base cases
    f(0) = 0, f(1) = 1, f(2) = 2
3. Define Recurrence Relation ()
    f(n) = f(n-2) + f(n-1)
4. Order of computation (bottom-up vs top-down)
    bottom-up
5. Location of the answer
    The answer will be in f(n)
"""


def climbStairs(n: int):
    """
    Time Complexity:
        O(n)
    Space Complexity:
        O(n)
    Args:
        n (int): _description_
    """

    lst = [0] * (n+1)

    lst[0], lst[1], lst[2] = 0, 1, 2

    for i in range(3, n+1):
        lst[i] = lst[i-2] + lst[i-1]
    return lst[n]


def climbStairsOptimalSpace(n: int):
    lst = [1, 1]

    for i in range(2, n+1):
        temp = lst[0]
        lst[0] = lst[1]
        lst[1] = lst[1] + temp
    return lst[1]


if __name__ == "__main__":
    print(climbStairs(3))
    print(climbStairs(4))
    print(climbStairs(5))
    print(climbStairsOptimalSpace(3))
    print(climbStairsOptimalSpace(4))
    print(climbStairsOptimalSpace(5))
