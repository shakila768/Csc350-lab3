"""CSC350 Lab 3: NumPy hands-on
Name:
Roll number:

Run this file after every task:  python lab3_numpy.py
Commit after every task (git add, git commit -m "Task N done").
"""
import numpy as np

print("NumPy version:", np.__version__)

# =====================================================================
# Task 1: Creating arrays (2 marks)
# =====================================================================
print("\n=== Task 1 ===")
# 1a. Make an array `temps` with this week's temperatures:
#     31, 28, 35, 30, 33, 29, 36
#     Print it, then print its dtype, shape and size.

# 1b. Make `months`: the numbers 1 to 12 as a 3 x 4 table
#     (use np.arange and reshape). Print it and its shape.

# 1c. Print an array of five zeros, and a 2 x 3 array of ones.

# 1d. Make a random generator with seed 42 and roll 10 dice
#     (whole numbers from 1 to 6). Print the result.
import numpy as np

print("--- Task 1 ---")

# 1a
temps = np.array([31, 28, 35, 30, 33, 29, 36])
print("1a temps:", temps)
print("dtype:", temps.dtype)
print("shape:", temps.shape)
print("size:", temps.size)

# 1b
months = np.arange(1, 13).reshape(3, 4)
print("\n1b months:")
print(months)
print("shape:", months.shape)

# 1c
five_zeros = np.zeros(5)
ones_2x3 = np.ones((2, 3))
print("\n1c Five zeros:", five_zeros)
print("2x3 ones:")
print(ones_2x3)

# 1d
rng = np.random.default_rng(42)
dice = rng.integers(1, 7, size=10)
print("\n1d Dice rolls (seed 42):", dice)


# =====================================================================
# Task 2: Indexing and masks (2.5 marks)  (use `temps` from Task 1)
# =====================================================================
print("\n=== Task 2 ===")
# 2a. Print the first and the last temperature.
# 2b. Print the weekend temperatures (the last two days).
# 2c. Print the temperatures in reverse order.
# 2d. Print only the hot days (above 32), how many there are,
#     and their average.  No loops!
# 2e. Print the temperatures from 29 to 33 (both included). Hint: &
# 2f. Use np.where to label each day "Hot" (33 or more) or "OK".
print("\n--- Task 2 ---")

# 2a - First and last
print("2a First:", temps[0], "Last:", temps[-1])

# 2b - Weekend (last two days)
print("2b Weekend:", temps[-2:])

# 2c - Reverse order
print("2c Reverse:", temps[::-1])

# 2d - Hot days (above 32), count and average - No loops!
hot_days = temps[temps > 32]
print("2d Hot days:", hot_days)
print("How many:", hot_days.size)
print("Average:", hot_days.mean())

# 2e - From 29 to 33 (both included)
print("2e 29 to 33:", temps[(temps >= 29) & (temps <= 33)])

# 2f - Label Hot (33 or more) or OK
labels = np.where(temps >= 33, "Hot", "OK")
print("2f Labels:", labels)

# =====================================================================
# Task 3: Gradebook (3 marks)
# Rows = students, columns = 4 subjects.
# =====================================================================
print("\n=== Task 3 ===")
names = np.array(["Ali", "Sara", "Usman", "Hina", "Bilal"])
marks = np.array([[78, 85, 62, 90],
                  [92, 88, 95, 79],
                  [45, 60, 48, 70],
                  [88, 94, 91, 85],
                  [65, 42, 75, 58]])
# 3a. Print the shape of marks.
# 3b. Print Usman's marks (one row) and all marks of subject 3
#     (column index 2).
# 3c. Print the average of each subject and the average of each student.
# 3d. Print the NAME of the student with the highest average.
#     Hint: argmax gives a position; use it on `names`.
# 3e. Subject weights are 0.3, 0.3, 0.2, 0.2. Print each student's
#     weighted total, rounded to 1 decimal. (Broadcasting!)
# 3f. Add 5 grace marks to everyone, but no mark may go above 100.
#     Print the new table. Hint: np.clip
# 3g. Print how many students failed (below 50) in each subject.
print("\n--- Task 3 ---")

# 3a - Shape of marks
print("3a Shape:", marks.shape)

# 3b - Usman ke marks (row 1) aur Subject 3 ke saare marks (column 2)
# Agar Usman ki row alag hai to number badal dena
print("3b Usman marks:", marks[1])
print("3b Subject 3 marks:", marks[:, 2])

# 3c - Har subject ka average aur har student ka average
print("3c Subject average:", marks.mean(axis=0))
print("3c Student average:", marks.mean(axis=1))

# 3d - Sab se zyada average wale student ka NAME
student_avg = marks.mean(axis=1)
top_student = np.argmax(student_avg)
print("3d Top student:", names[top_student])

# 3e - Weighted total, 1 decimal tak (Broadcasting)
weights = np.array([0.3, 0.3, 0.2, 0.2])
weighted_total = np.round((marks * weights).sum(axis=1), 1)
print("3e Weighted total:", weighted_total)

# 3f - 5 grace marks, lekin 100 se upar nahi
new_marks = np.clip(marks + 5, 0, 100)
print("3f New marks:")
print(new_marks)

# 3g - Har subject me kitne fail (below 50)
failed = (marks < 50).sum(axis=0)
print("3g Failed in each subject:", failed)

# =====================================================================
# Task 4: An image is an array (2.5 marks)
# A 5 x 5 grayscale image: 0 = black, 255 = white.
# =====================================================================
print("\n=== Task 4 ===")
img = np.array([[ 10,  10, 200,  10,  10],
                [ 10,  10, 220,  10,  10],
                [210, 230, 250, 230, 210],
                [ 10,  10, 220,  10,  10],
                [ 10,  10, 200,  10,  10]])
# 4a. Print the shape, the darkest value and the brightest value.
# 4b. Print the inverted image (white becomes black). Hint: 255 - ...
# 4c. Print a black-and-white version: 1 where the pixel is above 128,
#     else 0. Hint: a mask, then .astype(int)
# 4d. Print row 2 normalized to the range 0 to 1, rounded to 2 decimals.
# 4e. Print the 3 x 3 centre of the image (crop).
import numpy as np

print("\n--- Task 4 ---")
image = np.array([...])   # apna data / array yahan

# 4a - Shape, darkest aur brightest value
print("4a Shape:", image.shape)
print("4a Darkest:", np.min(image))
print("4a Brightest:", np.max(image))

# 4b - Inverted image
inverted = 255 - image
print("4b Inverted:")
print(inverted)

# 4c - Black and White image
bw = (image > 128).astype(int)
print("4c Black-and-white:")
print(bw)

# 4d - Row 2 ko 0 to 1 mein normalize karna
row2_norm = np.round(image[2] / 255.0, 2)
print("4d Row 2 normalized:", row2_norm)

# 4e - 3x3 Centre crop
centre = image[1:4, 1:4]
print("4e 3x3 Centre:")
print(centre)