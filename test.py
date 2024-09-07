import datetime

gtd = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=12)).timestamp()
goodTillDate = int(gtd * 1000)
print(goodTillDate)
