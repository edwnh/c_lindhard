CC = icx
CFLAGS += -static-intel -xcore-avx2 -qopenmp -qopenmp-link=static -qmkl=sequential
CFLAGS += -shared -fPIC -O3 -std=c17 -Wall -Wextra

all:
	${CC} ${CFLAGS} -o liblindhard.so lindhard.c
