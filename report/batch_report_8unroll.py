import subprocess
import shlex
import sys
import re

dims = [256, 512, 768, 1023, 1024, 1025, 2047, 2048, 2049]

for N in dims:
    print "Computing for "+str(N)
    mmpy_output = subprocess.Popen(shlex.split('./mmpy -n '+str(N)+' -r 1'), stdout=subprocess.PIPE).communicate()[0]
    start = mmpy_output.find('[')
    end = mmpy_output.find(' gflops')
    f = open("run_sorken_8unroll.txt", "a")
    f.write(str(N)+'-'+mmpy_output[start+1:end]+'\n')
    f.close()
