
def dict2list(dic:dict):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def count(filename):
    movie_count_map = {}
    line_count = 0
    with open(filename) as fin:
        for line in fin.readlines():
            items = line.split('::')
            if len(items) != 4 : continue
            line_count += 1
            movie_id = items[1]
            if movie_id not in movie_count_map:
                movie_count_map[movie_id] = 1
            else:
                tmp_count = movie_count_map[movie_id] + 1
                movie_count_map[movie_id] = tmp_count
    print(line_count)
    return movie_count_map

def loadMovieDict(movies_file):
    movies_dict = {}
    with open(movies_file) as fin:
        for line in fin.readlines():
            items = line.split('\t')
            if len(items) != 3 : continue
            movies_dict[items[0]] = items[1]
    return movies_dict

def userRating(ratings_file, movies_dict, user_id='1'):
    has_user = False
    user_movies = []
    with open(ratings_file) as fin:
        for line in fin.readlines():
            items = line.split('::')
            if len(items) != 4 : continue
            if items[0] == user_id:
                user_movies.append((movies_dict.get(items[1], "unk"), items[2]))
                has_user = True
            elif has_user : break
    return user_movies

ratings_file = '/home/ethan/Documents/ethanShare/kg rec joint/data/ml-1m/ratings.dat'
movies_file = '/home/ethan/Documents/ethanShare/kg rec joint/data/ml-1m/MappingMovielens2DBpedia-1.2.tsv'
""" n=10
movies_map = count(ratings_file)
sorted_movies = sorted(dict2list(movies_map), key=lambda x: x[1], reverse=True)
print(sorted_movies[:n]) """

movies_dict = loadMovieDict(movies_file)
user_movies = userRating(ratings_file, movies_dict, user_id='1')
out_str = '\n'.join([x[0].strip() for x in user_movies])
print(out_str)