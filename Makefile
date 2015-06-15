INC = -I. -I./inc 
LIBS = -lblas 

OPENMP = -fopenmp

CFLAGS += -g ${OPENMP} -std=c99 -pedantic -Wall -O3 ${INC}
LDFLAGS += -g ${OPENMP} ${LIBS}

CC = gcc-4.9 

SRC = eden.c
OBJ = ${SRC:.c=.o}

all: options eden

options:
	@echo eden build options:
	@echo "CFLAGS  = ${CFLAGS}"
	@echo "LDFLAGS = ${LDFLAGS}"
	@echo "CC      = ${CC}"

.c.o:
	@echo CC $<
	@${CC} -c ${CFLAGS} $<

eden: ${OBJ}
	@echo CC -o $@
	@${CC} -o $@ ${OBJ} ${LDFLAGS}

clean:
	@echo cleaning
	@rm -f eden ${OBJ} 

.PHONY: all options clean
