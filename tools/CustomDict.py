class CustomDict:

    def insert_value(self, k, v):
        try:
            self.d_results[k].append(v)
        except KeyError:
            inner_list=[]
            inner_list.append(v)
            self.d_results[k] = inner_list
            pass


    def __init__(self):
        self.d_results = dict()
        for i in range(0,101):
            self.d_results[i/100]=[]


    def fill_with_zeros(self):
        for k,v in self.d_results.items():
            if len(v)==0:
               self.d_results[k].append(0)