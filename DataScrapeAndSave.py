import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def main():
    # supaData()
    # exit(0)
    combineData()
    exit(0)
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    for pos in positions:
        for year in range(2010, 2017):
            inmates_list = []
            for i in range(1, 18):
                getDatar(i, str(year), pos, inmates_list)
            # After building up a dictionary with data for a position
            # through a year for all weeks, we create a dataframe using pandas
            df = pd.DataFrame(inmates_list)
            # we can export this data to any format your heart desires. I've like csv for this
            df.to_csv(
                "~/desktop/new-fantasyfootballdata/" + str(pos) + '_' + str(year) + '_' + "FantasyFootballData.csv")

    print("head\n", df.head());
    print("\nShape\n", df.shape)
    print("\n values\n", df.get_values())
    print("#" * 20, "\n items\n", df.items)
# Down below we'll add our inmates to this list


# We've now imported the two packages that will do the heavy lifting
# for us, reqeusts and BeautifulSoup. Pandas will come in handy later

# Let's put the URL of the page we want to scrape in a variable
# so that our code down below can be a little cleaner

def getDatar(week, year, position, inmates_list):
    # Our url takes in positions, years, and weeks as arguments, builds up a dictionary with data for each position
    # through each year for all weeks
    url_to_scrape = 'http://www.footballdb.com/fantasy-football/index.html?pos='\
                    +str(position)+'&yr='+str(year)+'&wk='+str(week)+'&rules=1'

    # Tell requests to retreive the contents our page (it'll be grabbing
    # what you see when you use the View Source feature in your browser)
    r = requests.get(url_to_scrape)

    # We now have the source of the page, let's ask BeaultifulSoup
    # to parse it for us.
    soup = BeautifulSoup(r.text, "html5lib")

    # BeautifulSoup provides nice ways to access the data in the parsed
    # page. Here, we'll use the select method and pass it a CSS style
    # selector to grab all the rows in the table (the rows contain the
    # inmate names and ages).

    for table_row in soup.select("table.statistics tr"):
        # Each tr (table row) has three td HTML elements (most people
        # call these table cels) in it (first name, last name, and age)
        cells = table_row.findAll('td')

        # Our table has one exception -- a row without any cells.
        # Let's handle that special case here by making sure we
        # have more than zero cells before processing the cells
        if len(cells) > 6: #why because the number 6 works. that's why
            first_name = cells[0].text.strip()
            name = first_name.split(',')[0]
            opponent = cells[1].text.strip()
            points_ff = cells[2].text.strip()

            # Let's add our inmate to our list in case
            # We do this by adding the values we want to a dictionary, and
            # appending that dictionary to the list we created above
            if (position in ['QB','RB','WR','TE']):
                points_ff = cells[2].text.strip()
                attempts_passing = cells[3].text.strip()
                completions_passing = cells[4].text.strip()
                yards_passing = cells[5].text.strip()
                td_passing = cells[6].text.strip()
                interceptions_passing = cells[7].text.strip()
                twoPoint_passing = cells[8].text.strip()

                attempts_rushing = cells[9].text.strip()
                yards_rushing = cells[10].text.strip()
                td_rushing = cells[11].text.strip()
                twoPoint_rushing = cells[12].text.strip()

                receptions_receiving = cells[13].text.strip()
                yards_receiving = cells[14].text.strip()
                td_receiving = cells[15].text.strip()
                twoPoint_receiving = cells[16].text.strip()
                lost_fumbles = cells[17].text.strip()
                td_fumbles = cells[18].text.strip()

                if (len(name) > 1 and len(opponent) > 0):
                        inmate = dict({'Name': name, 'Position': position, 'Opponent': opponent, 'Week': week,
                                       'Year': year, 'Points': points_ff, 'Pass Attempts': attempts_passing,
                                       'Pass Completions': completions_passing, 'Pass Yards': yards_passing,
                                       'Pass TD': td_passing, 'Pass Interceptions': interceptions_passing,
                                       'Pass 2PT': twoPoint_passing, 'Rush Attempts': attempts_rushing,
                                       'Rush Yards': yards_rushing, 'Rush TD': td_rushing,
                                       'Rush 2PT': twoPoint_rushing, 'Receptions': receptions_receiving,
                                       'Receiving Yards': yards_receiving, 'Receiving TD': td_receiving,
                                       'Receiving 2PT': twoPoint_receiving, 'Fumbles Lost': lost_fumbles,
                                       'Fumble TD': td_fumbles})
                        inmates_list.append(inmate)

            elif (position == 'K'):
                XPA = cells[3].text.strip()
                XPM = cells[4].text.strip()
                FGA = cells[5].text.strip()
                FGM = cells[6].text.strip()
                fiftyPlus = cells[7].text.strip()
                if (len(name) > 1 and len(opponent) > 0):
                    inmate = dict({'Name': name, 'Position': position, 'Opponent': opponent, 'Week': week,
                                   'Year': year, 'Points': points_ff, 'XPA': XPA, 'XPM': XPM, 'FGA': FGA,
                                   'FGM':FGM, '50+':fiftyPlus})

                    inmates_list.append(inmate)
                #For the Defense/SpecialTeams
            elif (position == 'DST'):
                opponent = cells[1].text.strip()
                points_ff = cells[2].text.strip()
                sack = cells[3].text.strip()
                interceptions = cells[4].text.strip()
                safety = cells[5].text.strip()
                fr = cells[6].text.strip()
                blocks = cells[7].text.strip()
                td = cells[8].text.strip()
                pa = cells[9].text.strip()
                py = cells[10].text.strip()
                ry = cells[11].text.strip()
                ty = cells[12].text.strip()
                if (len(name) > 1 and len(opponent) > 0):
                    inmate = dict({'Team': name, 'Opponent': opponent,'Position': position,  'Week': week,
                                   'Year': year, 'Points': points_ff, 'Sack': sack, 'Interception': interceptions,
                                   'Safety': safety, 'Fumble Returns': fr, 'Blocks':blocks, 'Touchdowns': td,
                                   'Point After': pa, 'Pass Yards Allowed': py, 'Rush Yards Allowed': ry,
                                   'Total Yards': ty})
                    inmates_list.append(inmate)
            else:
                exit("Unsure which position you're attempting to get data for")

        # The byes we originally received from BeautifulSoup is a
        # string. We need it to be a number so that we can compare
        # it easily. Let's make it an integer.
        # passPoints = float(inmate['Pass Points'])

def combineData():
    positions = ['K']
    for i in range(2010, 2017):
        os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets')
        for pos in positions:
            os.chdir(pos)
            frames = []
            for file in os.listdir(os.getcwd()):
                tempDF = pd.read_csv(file)
                tempDF.set_index(['Year', 'Week','Name'], inplace=True)
                frames.append(tempDF)
            a = pd.concat(frames)
            try:
                a.drop('Unnamed: 0.1', axis=1, inplace=True)
            except: pass
            try:
                a.drop('Unnamed: 0', axis=1, inplace=True)
            except: pass
            print(pos, '\n', a.head())
            a.to_csv('/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets/' + pos + 'AllYears.csv')
            os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets')

def supaData():
    files = ['qbAllYears.csv', 'rbAllYears.csv', 'wrAllYears.csv', 'teAllYears.csv']

    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/Data')
    frames = []
    for file in files:
        frames.append(pd.read_csv(file))
    a = pd.concat(frames)
    a.drop('Unnamed: 0', axis=1, inplace=True)
    a.to_csv('/Users/nickdugal/Documents/Fantasy-Football/data/AllPositionsAllYears.csv', index=None)
    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data')

if __name__ == "__main__": main()