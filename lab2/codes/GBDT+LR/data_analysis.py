from collections import defaultdict

# 读取文件并处理数据
def read_attribute_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            attr1 = int(parts[1]) if parts[1] != 'None' else -1
            attr2 = int(parts[2]) if parts[2] != 'None' else -1
            data.append((item_id, attr1, attr2))
    return data

# 统计属性类型出现次数
def count_attributes(data):
    attribute_count = defaultdict(int)
    for item_id, attr1, attr2 in data:
        if attr1 is not None:
            attribute_count[attr1] += 1
        if attr2 is not None:
            attribute_count[attr2] += 1
    return attribute_count

# 计算总共出现多少个属性类型、平均每个属性出现多少次、最少出现和最多出现的属性类型次数
def analyze_attributes(attribute_count):
    total_attributes = len(attribute_count)
    total_occurrences = sum(attribute_count.values())
    if total_attributes > 0:
        avg_occurrences = total_occurrences / total_attributes
    else:
        avg_occurrences = 0
    min_occurrences = min(attribute_count.values(), default=0)
    max_occurrences = max(attribute_count.values(), default=0)
    return total_attributes, avg_occurrences, min_occurrences, max_occurrences


def main(filepath):
    data = read_attribute_data(filepath)
    attribute_count = count_attributes(data)
    total_attributes, avg_occurrences, min_occurrences, max_occurrences = analyze_attributes(attribute_count)
    
    print(f"There are a total of {total_attributes} attribute types")
    print(f"On average, each attribute appears {avg_occurrences:.2f} times")
    print(f"The least frequent attribute type appeared {min_occurrences} times")
    print(f"The most frequent attribute type appeared {max_occurrences} times")


if __name__ == "__main__":
    filepath = "../data/itemAttribute.txt"
    main(filepath)
