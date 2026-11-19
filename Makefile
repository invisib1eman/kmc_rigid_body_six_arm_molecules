PROG1 = NANOROD

SRC1 = main.cpp system.cpp utils.cpp mc.cpp quarternion.cpp

OBJS1 = ${SRC1:.cpp=.o}

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++11
GSL_CFLAGS := $(shell pkg-config --cflags gsl 2>/dev/null)
GSL_LIBS := $(shell pkg-config --libs gsl 2>/dev/null)
CPPFLAGS += $(GSL_CFLAGS)
LDLIBS += -lboost_program_options $(if $(GSL_LIBS),$(GSL_LIBS),-lgsl -lgslcblas -lm)
# macOS (Homebrew)
ifneq ($(wildcard /opt/homebrew/include),)
CPPFLAGS += -I/opt/homebrew/include
LDFLAGS  += -L/opt/homebrew/lib
endif

# Great Lakes (Boost module)
ifneq ($(EBROOTBOOST),)
CPPFLAGS += -I$(EBROOTBOOST)/include
LDFLAGS  += -L$(EBROOTBOOST)/lib
else ifneq ($(BOOST_ROOT),)
CPPFLAGS += -I$(BOOST_ROOT)/include
LDFLAGS  += -L$(BOOST_ROOT)/lib
endif

DEPS1 = ${OBJS1:.o=.d}

all: $(PROG1) 

$(PROG1):  $(OBJS1)
	 $(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@ 

%.o:  %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c -o $@ $<

clean: 
	rm -f *.o *.d

distclean:
	rm -f $(PROG1) *.o *.d *.debug *.txt *.log *.mol2 *.lammpstrj

rerunclean:
	rm -f *.debug *.txt *.log *.mol2 *.lammpstrj

-include $(DEPS1)

.PHONY: all clean distclean rerunclean
