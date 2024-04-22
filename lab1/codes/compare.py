import os

def compare_files(file1_path, file2_path):
    file1_name, file1_ext = os.path.splitext(os.path.basename(file1_path))
    file2_name, file2_ext = os.path.splitext(os.path.basename(file2_path))

    output_file_name = f"{file1_name}_{file2_name}_comparison{file1_ext}"
    output_file_path = os.path.join(os.path.dirname(file1_path), output_file_name)

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(output_file_path, 'w') as output_file:
        for line1, line2 in zip(f1, f2):
            num1_1, num1_2 = map(float, line1.strip().split())
            num2_1, num2_2 = map(float, line2.strip().split())
            result1 = num1_1 - num2_1
            result2 = num1_2 - num2_2

            output_file.write(f"{result1} {result2}\n")

if __name__ == "__main__":
    file1_path = input("Enter path for the first file: ")
    file2_path = input("Enter path for the second file: ")

    compare_files(file1_path, file2_path)
