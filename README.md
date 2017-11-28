# Fantasy-Football
The following features are part of the standard fantasy football scoring algorithm
For each, we need to identify which other features positively correlate, negatively correlate, or don't correlate
Correlation is typically a good starting point for exploring sources of causation, though always remember that **correlation** and **causation** aren't correlated
*Idea: collaborative filtering predict stats->predict score -> stats -> score ->...till convergence

## Features and their Fantasy Football value - [espn standard](http://games.espn.com/ffl/resources/help/content?name=scoring-formats) 
PASSING   | POINT RATIO
--------- | -----------
TD | 1 : 4
Yards | 25 : 1
2PT Conversion | 1 : 2
Interceptions | 1 : -2

Rushing   | POINT RATIO
--------- | -----------
TD | 1 : 6
Yards | 10 : 1
2PT Conversion | 1 : 2

Receiving | POINT RATIO
--------- | -----------
TD | 1 : 6
Yards | 10 : 1
2PT Conversion | 1 : 2

Misc Offense | POINT RATIO
------------ | -----------
Kickoff Return TD | 1 : 6
Punt Return TD | 1 : 6
Fumble Recovered for TD | 1 : 6
Fumble Lost | 1 : -2

Kicking | POINT RATIO
------- | -----------
FG [50,∞) | 1 : 5
FG [40,49] | 1 : 4
FG (-∞,39] | 1 : 3
PAT Made | 1 : 1
FG Missed (-∞,∞) : -1

Punting | POINT RATIO
------- | -----------
Not used | in Standard game

