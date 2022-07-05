import time

class Timer(object):
    """docstring for Timer"""
    def __init__(self, class_name=[]):
        super(Timer, self).__init__()
        
        self.duration = {}
        self.start_time = {}
        self.names=[]
        for name in class_name:
            self.duration[name] = 0.0
            self.start_time[name] = 0.0
            self.names.append(name)


    def start(self,name):
        if name not in self.duration:
            print('No this key')
            return False

        self.start_time[name] = time.time()
        return True

    def end(self,name):
        if name not in self.duration:
            print('No this key')
            return None

        time_pass = time.time() - self.start_time[name]
        self.duration[name] = self.duration[name]+ time_pass
        return time_pass

    def result(self,ratio=False):
        if not ratio:
            for name in self.names:
                print(name, self.duration[name])
        else:
            total = 0.
            for name in self.duration:
                total+=self.duration[name]
            for name in self.names:
                #total+=self.duration[name]
                print(name, '%.f%%'%(self.duration[name]/total*100))
        return

if __name__ == '__main__':
    timer = Timer(['sb','cao'])
    timer.start('sb')
    timer.end('sb')

    timer.start('cao')
    timer.end('cao')

    timer.result()
