bitops.o: bitops.c bitops.h
libnuma.o: libnuma.c numa.h numaif.h numaint.h bitops.h util.h affinity.h
distance.o: distance.c numa.h numaint.h bitops.h
memhog.o: memhog.c numa.h numaif.h util.h
numactl.o: numactl.c numa.h numaif.h numaint.h bitops.h util.h shm.h
numademo.o: numademo.c numa.h
numamon.o: numamon.c
shm.o: shm.c numa.h numaif.h numaint.h bitops.h util.h shm.h
stream_lib.o: stream_lib.c stream_lib.h
stream_main.o: stream_main.c numa.h numaif.h util.h stream_lib.h
util.o: util.c numa.h numaif.h util.h bitops.h
mt.o: mt.c mt.h
clearcache.o: clearcache.c clearcache.h
distance.o: test/distance.c numa.h
ftok.o: test/ftok.c
getnodemask.o: test/getnodemask.c numa.h
migrate_pages.o: test/migrate_pages.c numa.h
mynode.o: test/mynode.c numa.h numaif.h
node-parse.o: test/node-parse.c numa.h util.h
nodemap.o: test/nodemap.c numa.h bitops.h
pagesize.o: test/pagesize.c
prefered.o: test/prefered.c numa.h numaif.h
randmap.o: test/randmap.c numa.h numaif.h
realloc_test.o: test/realloc_test.c numa.h numaif.h
tbitmap.o: test/tbitmap.c numa.h
tshared.o: test/tshared.c numa.h numaif.h
tshm.o: test/tshm.c numaif.h
sysfs.o: sysfs.c numa.h numaint.h bitops.h
numastat.o: numastat.c
