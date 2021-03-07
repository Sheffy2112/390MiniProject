# Python3 code to demonstrate
# to count words in string
# using split()

# initializing string
test_string = "Geeksforgeeks is best Computer Science Portal"

# printing original string
print ("The original string is : " + test_string)

# using split()
# to count words in string
res = len(test_string.split())

# printing result
print ("The number of words in string are : " + str(res))
print(type(res))
