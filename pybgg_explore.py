import json
from libbgg.apiv2 import BGG

conn = BGG()

# will capture all description text for a game
# stat_results = conn.boardgame(169786 , stats=True)
# description = stat_results['items']['item']['description']['TEXT']

#was thinking this was necessary, but found the above is accessible via
#standard dictionary style access
# scythe = json.dumps(results)
# scythe_load = json.loads(scythe)

#this will get every single rating and comment if they made one
#if the page specified does not exist a value error is raised, thus probably a
#good way to end a loop
comment_results = conn.boardgame(169786 , ratingcomments=True, page=100, pagesize=100)
print(comment_results['items']['item']['comments']['comment'])
