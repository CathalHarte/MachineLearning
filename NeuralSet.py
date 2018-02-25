class LearningSet:
    def __init__(self):
        self.x = 0
        self.y = 0
    
    def compare(self,image):
        sx = self.x
        sy = self.y
        self.x = max([self.x, image.shape[0]])
        self.y = max([self.x, image.shape[1]])
        if sx == self.x & sy == self.y:
            return True
        else:
            return False
        
    
    def prepareBatch(self, Images=self.Batch, classifier=self.label):
        # In this function, what is Images?
        # Classifier is an array of number classifications, if there are 
        # 25 distinct classifiers, then we find at each index a number between 0
        # and 24
        l = len(Images)
        self.label = np.zeros((l,2))
        self.Batch = np.zeros((l,self.x,self.y,1))
        for i in range(l):
            self.label[i,classifier[i]] = 1
            x = Images[i].shape[0]
            y = Images[i].shape[1]
            self.Batch[i,0:x,0:y,0] = Images[i]
    
    def append(self, a):
        if not compare(self,a):
            self.prepareBatch()
            