INCS = -I. -I/include
LIBS = 

CFLAGS += -g -fopenmp -std=c99 -pedantic -Wall -Os ${INC}
LDFLAGS += -g -fopenmp ${LIBS}

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
