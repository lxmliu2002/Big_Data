import statistics
from collections import defaultdict

def read_data(file_path):
    user_ratings = defaultdict(list)
    item_ratings = defaultdict(list)
    max_user_id = 0
    max_item_id = 0
    
    try:
        with open(file_path, 'r') as f:
            while True:
                user_line = f.readline()
                if not user_line:
                    break
                user_id, num_ratings = map(int, user_line.strip().split('|'))
                max_user_id = max(max_user_id, user_id)
                for _ in range(num_ratings):
                    item_line = f.readline()
                    item_id, rating = map(float, item_line.strip().split())
                    max_item_id = max(max_item_id, item_id)
                    user_ratings[user_id].append(rating)
                    item_ratings[item_id].append(user_id)
                    
        return user_ratings, item_ratings, max_user_id, max_item_id
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return {}, {}, 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}, {}, 0, 0

def calculate_statistics(user_ratings, item_ratings):
    num_users = len(user_ratings)
    num_items = len(item_ratings)
    total_ratings = sum(len(ratings) for ratings in user_ratings.values())
    average_rating = statistics.mean(rating for ratings in user_ratings.values() for rating in ratings)
    
    return num_users, num_items, total_ratings, average_rating

if __name__ == '__main__':
    file_path = './data/train.txt'
    user_ratings, item_ratings, max_user_id, max_item_id = read_data(file_path)
    num_users, num_items, total_ratings, average_rating = calculate_statistics(user_ratings, item_ratings)
    print("Number of users:", num_users)
    print("Number of items rated:", num_items)
    print("Total number of ratings:", total_ratings)
    print("Average rating:", average_rating)
    print("Max user ID:", max_user_id)
    print("Max item ID:", max_item_id)
