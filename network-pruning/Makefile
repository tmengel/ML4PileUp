CXX=g++
CXX_COMMON=-O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread
CXX_SHARED=-shared
CXX_SONAME=-Wl,-soname,

LIB_SUFFIX=.so
LD = g++

CXX_ROOT = $(shell root-config  --cflags --libs)


CXX_COMMON:= $(CXX_COMMON) $(GZIP_LIB) $(CXX_ROOT)

# Targets
BINS = $(patsubst %.cpp, %, $(wildcard *.cpp))
	
default : $(BINS)

$(BINS): % : %.cpp
	@echo -n "Building $@ ... "
	$(CXX) $<  -o $@ -w  $(CXX_COMMON)
	@echo "Done"

clean: 
	rm -f *.o	$(BINS)

jobclean:
	rm *.o* *.e*
