CFLAGS=`pkg-config opencv --cflags` -std=gnu++0x -pthread
LIBS=`pkg-config opencv --libs`

all: run

run: tracker
	./tracker ../printerManualRecord.mp4 cascades/square2/cascade.xml

time: tracker
	time ./tracker ~/Desktop/Tardis2.mp4 cascades/square2/cascade.xml

profile: trackerProfile
	./tracker ~/Desktop/Tardis2.mp4 cascades/square2/cascade.xml

tracker: tracker.cpp TagRegion.cpp
	g++ $(CFLAGS) -g -rdynamic tracker.cpp TagRegion.cpp -o tracker $(LIBS)

trackerProfile: tracker.cpp TagRegion.cpp
	g++ $(CFLAGS) -pg -g tracker.cpp TagRegion.cpp -o tracker $(LIBS)
