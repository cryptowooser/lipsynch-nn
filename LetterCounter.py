import os

def count_letters(directory):
    # Dictionary to store count of each letter A through I.
    count = {chr(i): 0 for i in range(65, 74)}
    count.update({chr(i): 0 for i in range(97, 106)}) # lowercase a-i

    # Text file extensions to be considered.
    text_extensions = ['.txt']

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in text_extensions):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        for char in line:
                            if char in count:
                                count[char] += 1

    return count

def main():
    directory = 'texts'  # Insert your directory here.
    result = count_letters(directory)

    total_count = sum(result.values())

    print(f"The letters A through I appear {total_count} times in the text files.")
    print("Detailed count per letter:")
    for letter, count in result.items():
        print(f"{letter}: {count}")

if __name__ == "__main__":
    main()
