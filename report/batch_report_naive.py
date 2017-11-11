import subprocess
import shlex
import sys
import re

dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N = str(sys.argv[1])

for x in dims:
    for y in dims:
        if x*y <= 1024:
            print "Computing for "+str(x)+" and "+str(y)
            mmpy_output = subprocess.Popen(shlex.split('./mmpy -n '+N+' -x '+str(x)+' -y '+str(y)+' -r 10'), stdout=subprocess.PIPE).communicate()[0]
            start = mmpy_output.find('[')
            end = mmpy_output.find(' gflops')
            f = open("run_sorken_"+N+"_naive.txt", "a")
            f.write(str(x)+'-'+str(y)+'-'+mmpy_output[start+1:end]+'\n')
            f.close()
