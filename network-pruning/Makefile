
-include Makefile.inc


BINS = $(patsubst %.cpp, %, $(wildcard *.cpp))
		


CXX = g++
CCFLAGS = $(shell root-config --cflags) #-ggdb 

CXX_PYTHIA= $(PYTHIA8_INCLUDE) $(PYTHIA8_LIB) -lpythia8 -ldl
CXX_COMMON:=-I$(PREFIX_INCLUDE) $(CXX_COMMON) $(GZIP_LIB)
CXX_COMMON+= -L$(PREFIX_LIB) -Wl,-rpath,$(PREFIX_LIB) -ltenngen200 -ldl


LD = g++
LDFLAGS = 

LIBS = $(shell root-config --libs) -lMLP -lMinuit -lTreePlayer -lTMVA -lTMVAGui -lXMLIO  -lMLP -lm
CXX_ROOT = $(ROOT_LIB) $(shell root-config --cflags --glibs)
default : $(BINS)

$(BINS): % : %.cpp 
	@echo -n "Building $@ ... "
	$(CXX)  $(CCFLAGS) $<  -o $@ -w  $(CXX_COMMON) $(CXX_PYTHIA) $(CXX_ROOT) $(FASTJET3_INCLUDE) $(FASTJET3_LIB) $(shell $(FASTJET3_CONFIG)/fastjet-config --cxxflags --libs)
	@echo "Done"


clean: 
	rm -f *.o	$(BINS)

jobclean:
	rm *.o* *.e*
