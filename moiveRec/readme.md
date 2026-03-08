MovieLens 1M 数据集：包含6000名用户对3900部电影的100万条评分数据。

用户信息文件（Users.dat），包含用户的基本信息，如用户ID、性别、年龄、职业等。

• UserID：用户唯一标识。从1~6040， 代表了6040个MovieLens用户
• Gender：性别（M表示男性，F表示女性）。
• Age：用户年龄，分成了7组
    • 1: "Under 18"
    • 18: "18-24"
    • 25: "25-34"
    • 35: "35-44"
    • 45: "45-49"
    • 50: "50-55"
    • 56: "56+"

• Occupation：用户职业，如学生、教师、工程师等。
    • 0: "other" or not specified
    • 1: "academic/educator"
    • 2: "artist"
    • 3: "clerical/admin"
    • 4: "college/grad student"
    • 5: "customer service"
    • 6: "doctor/health care"
    • 7: "executive/managerial"
    • 8: "farmer"
    • 9: "homemaker"
    • 10: "K-12 student"
    • 11: "lawyer"
    • 12: "programmer"
    • 13: "retired"
    • 14: "sales/marketing"
    • 15: "scientist"
    • 16: "self-employed"
    • 17: "technician/engineer"
    • 18: "tradesman/craftsman"
    • 19: "unemployed"
    • 20: "writer"


• Zip-code：用户所在地区的邮政编码。


电影信息文件（Movies.dat），MovieID::Title::Genres。


• MovieID：电影唯一标识。从1~3952， 代表了3952部电影
• Title：电影标题，通常包含电影名称和发行年份。
• Genres：电影题材由竖线分开， 包括动作喜剧等18种电影类型，如“Action|Comedy”。
    • Action
    • Adventure
    • Animation
    • Children's
    • Comedy
    • Crime
    • Documentary
    • Drama
    • Fantasy
    • Film-Noir
    • Horror
    • Musical
    • Mystery
    • Romance
    • Sci-Fi
    • Thriller
    • War
    • Western






评分文件（Ratings.dat），UserID::MovieID::Rating::Timestamp




• UserID：用户唯一标识。
• MovieID：电影唯一标识。
• Rating：用户对电影的评分，通常为1到5的整数。
• Timestamp：评分的时间戳，表示自1970年1月1日以来的秒数。