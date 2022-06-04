from utils import get_class_infors, modify_infors, get_all_infors

if __name__ == '__main__':
    hotel, scenic, travel, food, news = get_class_infors()
    hotel, scenic, travel, food, news = modify_infors(hotel, scenic, travel, food, news)
    all_df = get_all_infors(hotel, scenic, travel, food, news)
    print(all_df.head())
    print(all_df['text'].values[0])