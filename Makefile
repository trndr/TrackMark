CFLAGS=`pkg-config opencv --cflags` -std=gnu++0x -pthread
LIBS=`pkg-config opencv --libs`

all: run

run: tracker
	./tracker ../Tardis.mp4 cascades/square2/cascade.xml

time: tracker
	time ./tracker ../Tardis2.mp4 cascades/square2/cascade.xml

profile: trackerProfile
	./tracker ../Tardis2.mp4 cascades/square2/cascade.xml

tracker: tracker.cpp TagRegion.cpp
	g++ $(CFLAGS) -Wall -rdynamic tracker.cpp TagRegion.cpp -o tracker $(LIBS)

invProj: invProj.cpp 
	g++ $(CFLAGS) -rdynamic invProj.cpp -o invProj $(LIBS)

trackerProfile: tracker.cpp TagRegion.cpp
	g++ $(CFLAGS) -pg -g tracker.cpp TagRegion.cpp -o tracker $(LIBS)
