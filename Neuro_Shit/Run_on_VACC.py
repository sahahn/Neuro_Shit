import os
import random
import shutil


class VACC():

    def __init__(self, base_script_contents='', command='', pool_mem=True,
                 ppn='2', mem='6gb', vmem='8gb', walltime='6:00:00',
                 name='script', fs_import=False):

        self.key = str(random.random())
        self.contents = base_script_contents
        self.command = command
        self.make_base_script()

        self.pool_mem = pool_mem
        self.ppn = ppn
        self.mem = mem
        self.vmem = vmem
        self.walltime = walltime
        self.name = name
        self.fs_import = fs_import

        self.make_vacc_script()
        self.run_script()

    def make_base_script(self):

        if len(self.contents) > 0:

            with open('temp' + self.key + '.py', 'w') as f:
                for line in self.contents:
                    f.write(line)
                    f.write('\n')

    def make_vacc_script(self):

        with open('temp' + self.key + '.script', 'w') as f:

            if self.pool_mem:
                f.write('#PBS -qpoolmemq')
                f.write('\n')

            f.write('#PBS -l nodes=1:ppn=' + self.ppn)
            f.write(',mem=' + self.mem + ',vmem=' + self.vmem)
            f.write('\n')

            f.write('#PBS -l walltime=' + self.walltime)
            f.write('\n')

            f.write('#PBS -N ' + self.name)
            f.write('\n')

            f.write('#PBS -j oe\n')

            if self.fs_import:
                f.write('source /users/n/a/nallgaie/.bashrc\n')
                f.write('export FREESURFER_HOME=/gpfs1/arch/x86_64-rhel7/freesurfer-5.3.0-HCP/\n')
                f.write('source /gpfs1/arch/x86_64-rhel7/freesurfer-5.3.0-HCP/SetUpFreeSurfer.sh\n')
                f.write('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/n/a/nallgaie/lib:/lib64\n')

            f.write('cd ')
            f.write(os.getcwd() + ' ')
            f.write('\n')

            if len(self.contents) > 0:
                f.write('python temp' + self.key + '.py')
            else:
                f.write(self.command)

            f.write('\n')

    def run_script(self):
        os.system('qsub temp' + self.key + '.script')
        os.remove('temp' + self.key + '.script')
