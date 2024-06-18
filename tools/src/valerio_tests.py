import numpy as np
import distance as distance

class ValUnitEuclidian(object):

    def __init__(self,wlen:int,**kwargs) -> None:
        """Initialization

        Args:
            wlen (int): sliding window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray)->None: 
        """Initialize the distance accordingly to the signal considered. 
        Compute the first line of the crossdistance matrix and the elementary elements required for reccurssivity.

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)

        """
        self.signal_=signal
        self._first_elemntary()
        
    def _first_elemntary(self)->None:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """        
        self.first_dot_product_= np.convolve(self.signal_[:self.wlen:][::-1],self.signal_,'valid')

        means =np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_= means[self.wlen:]-means[:-self.wlen]
    
        stds= np.zeros(len(self.signal_)+1)
        stds[1:] = np.cumsum(self.signal**2)/self.wlen
        stds = stds[self.wlen:]-self[:-self.wlen]
        self.stds_=np.sqrt(stds-self.means_**2)

    def first_line(self,i:int)->np.ndarray: 
        """Compute the line of the crossdistance matrix at index i 

        Args:
            i (int): line position

        Returns:
            np.ndarray: line i of the crossdistance matrix
        """
        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_= np.convolve(self.signal[i:self.w+i][::-1])
        else:
            self.dot_product_= np.copy(self.first_dot_product_)

        dist=(self.dot_product_-self.wlen*self.means_[i]*self.means_)/(self.wlen*self.stds_[i]*self.stds_)

        return np.sqrt(2*self.wlen*(1-dist))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:]=self.dot_product_[:-1]-self.signal[self.idx_]*self.signal[:-self.wlen] + self.signal[self.wlen+self.idx_]*self.signal[self.wlen:]
        self.idx_ +=1
        self.dot_product_[0]=self.first_dot_product_[self.idx_]
        dist=(self.dot_product_-self.wlen*self.means_[self.idx]*self.means_)/(self.wlen*self.stds_[self.idx]*self.stds_)
        return np.sqrt(2*self.wlen*(1-dist))
    
    def individual_distance(self,i,j):
        a=(self.signal[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b=(self.signal[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))
    
import numpy as np
import distance as distance

class ValUnitEuclidian(object):

    def __init__(self,wlen:int,**kwargs) -> None:
        """Initialization

        Args:
            wlen (int): sliding window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray)->None: 
        """Initialize the distance accordingly to the signal considered. 
        Compute the first line of the crossdistance matrix and the elementary elements required for reccurssivity.

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)

        """
        self.signal_=signal
        self._first_elemntary()
        
    def _first_elemntary(self)->None:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """        
        self.first_dot_product_= np.convolve(self.signal_[:self.wlen:][::-1],self.signal_,'valid')

        means =np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_= means[self.wlen:]-means[:-self.wlen]
    
        stds= np.zeros(len(self.signal_)+1)
        stds[1:] = np.cumsum(self.signal**2)/self.wlen
        stds = stds[self.wlen:]-self[:-self.wlen]
        self.stds_=np.sqrt(stds-self.means_**2)

    def first_line(self,i:int)->np.ndarray: 
        """Compute the line of the crossdistance matrix at index i 

        Args:
            i (int): line position

        Returns:
            np.ndarray: line i of the crossdistance matrix
        """
        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_= np.convolve(self.signal[i:self.w+i][::-1])
        else:
            self.dot_product_= np.copy(self.first_dot_product_)

        dist=(self.dot_product_-self.wlen*self.means_[i]*self.means_)/(self.wlen*self.stds_[i]*self.stds_)

        return np.sqrt(2*self.wlen*(1-dist))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:]=self.dot_product_[:-1]-self.signal[self.idx_]*self.signal[:-self.wlen] + self.signal[self.wlen+self.idx_]*self.signal[self.wlen:]
        self.idx_ +=1
        self.dot_product_[0]=self.first_dot_product_[self.idx_]
        dist=(self.dot_product_-self.wlen*self.means_[self.idx]*self.means_)/(self.wlen*self.stds_[self.idx]*self.stds_)
        return np.sqrt(2*self.wlen*(1-dist))
    
    def individual_distance(self,i,j):
        a=(self.signal[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b=(self.signal[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))
    
class ValmodVal(object):

    def __init__(self,n_patterns:int,min_wlen:int,max_wlen:int,p:int,distance_name:str,distance_params =dict(),step=1,radius_ratio=3,n_jobs=1)-> None:
        """Initialization

            Args:
                n_patterns (int): Number of neighbors
                min_wlen (int): Minimum window length
                max_wlen (int): Maximum window length
                p(int): minimal number of distances computed in any cases
                distance_name (str): name of the distance
                distance_params (dict, optional): additional distance parameters. Defaults to dict().
                step (dict, optional): wlen step. Defaults to 1.
                radius_ratio (float): radius as a ratio of min_dist. 
                n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_patterns = n_patterns
        self.radius_ratio = radius_ratio
        self.min_wlen = min_wlen
        self.max_wlen = max_wlen
        self.p = p 
        self.distance_name = distance_name
        self.step =step
        self.distance_params = distance_params
        self.n_jobs = n_jobs

    def CompLB(self,idx:int,next_idx:int,i:int)->np.ndarray:
        """Compute the LowerBound of the distance between Ti,l+k and Tj,l+k for all j
            Args:
                idx(int) : window length index
                next_idx(int) : next window length index
                i (int): considered subsequence
            returns: 
                np.ndarray: lower bounds array
        """
        wlen = self.wlens_[idx]
        next_wlen=self.wlens_[next_idx]
        next_nDP=self.signal_.shape[0]-next_wlen+1
        next_non_overlap_mask=np.arange(max(0,i-next_wlen+1), min(next_nDP,i+next_wlen))
        
        q=(self.distance_[idx].dot_product_ -self.distance_[idx].means_[i]*self.distance_[idx].means_)/(self.distance_[idx].stds_[i]*self.distance_[idx].stds_)
        q=q[:next_nDP]
        q[next_non_overlap_mask]= np.inf
        
        LB=np.zeros(next_nDP)
        mask= q<=0
        LB[mask]=np.sqrt(wlen)*(self.distance_[idx].stds_[:next_nDP][mask]/self.distance_[next_idx].stds_[mask])
        LB[~mask]=np.sqrt(wlen*(1-q[~mask]**2))*(self.distance_[idx].stds_[:next_nDP][~mask]/self.distance_[next_idx].stds_[~mask])

        return LB
    
    def ComputeMatrixProfile(self,idx:int)-> tuple:
        """Compute the Matrix Profile and the LowerBound for the length corresponding to idx
            Args:
                idx(int): window length index
            Returns: 
                MP(np.ndarray): MatrixProfile
                IP(np.ndarray: Index Profile
                listDP(list of np.ndarray): list containing for each i the successive informations:
                    -the indexes of the p minimum Dij
                    -the corresponding distances
                    -the corresponding LB
                    -the corresponding dot_products
                """
        MP=np.zeros(self.nDP)
        IP=np.zeros(self.nDP).astype(int)
        listDP=[]

        line=self.distance_[idx].first_line(0)
        non_overlap_mask = np.arange(0, min(self.nDP,self.wlens_[idx]))
        line[non_overlap_mask] = np.inf
        MP[0]=np.min(line)
        IP[0]=np.argmin(line)
        #we don't need the lower bound for the last index 
        if idx+1< self.wlens_.shape[0]:
            LB=self.CompLB(idx,idx+1,0)
            idx_sort=np.argsort(LB)[:self.p]
            trunc_dist=line[idx_sort]
            trunc_LB=LB[idx_sort]
            trunc_dot_prod=self.distance_[idx].dot_product_[idx_sort]

            listDP.append(DP(idx_sort, trunc_dist, trunc_LB, trunc_dot_prod))

        for i in range(1,self.nDP):
            line=self.distance_[idx].next_line()
            non_overlap_mask = np.arange(max(0,i-self.wlens_[idx]+1), min(self.nDP,i+self.wlens_[idx]))
            line[non_overlap_mask] = np.inf
            MP[i]=np.min(line)
            IP[i]=np.argmin(line)
            if idx+1< self.wlens_.shape[0]:
                LB=self.CompLB(idx,idx+1,i)
                idx_sort=np.argsort(LB)[:self.p]
                trunc_dist=line[idx_sort]
                trunc_LB=LB[idx_sort]
                trunc_dot_prod=self.distance_[idx].dot_product_[idx_sort]

                listDP.append(DP(idx_sort, trunc_dist, trunc_LB, trunc_dot_prod))

        return MP, IP, listDP

    def updateDistAndLB(self, idx:int, next_idx:int, i:int, j:int,dot_product:np.ndarray, LB:np.ndarray)-> tuple:
        #should be vectorized to be more efficient
        """Update the distance and lowerbound for the sequences i and j from a length to the next one 
            Args:
                idx(int): window length index
                next_idx(int): next window length index
                i(int): offset of the first subsequence (the one for which we don't know Ti,l+k)
                j(int): offset of the second subsequence (the one for which we know Tj,l+k)
                dot_product(np.ndarray): dot product QTi,j for len wlen
                LB(np.ndarray): lower bound of Di,j for len l
            Returns:
                new_distance(np.ndarray): updated Di,j for len l+k
                new_LB(np.ndarray):updated LB for len l+k
        """
        wlen=self.wlens_[idx]
        newlen=self.wlens_[next_idx]
        new_dot_product=dot_product+np.dot(self.signal_[i+wlen:i+newlen],self.signal_[j+wlen:j+newlen])
        #we check if the increase of the length don't create an overlap
        
        if np.abs(i-j)<newlen:
            new_dist=np.inf
            new_LB=np.inf
        else:    
            new_dist=(new_dot_product-newlen*self.distance_[next_idx].means_[i]*self.distance_[next_idx].means_[j])/(newlen*self.distance_[next_idx].stds_[i]*newlen*self.distance_[next_idx].stds_[j])
            new_dist=np.sqrt(2*newlen*(1-new_dist))
        
            new_LB= LB * self.distance_[idx].stds_[j]/self.distance_[next_idx].stds_[j]

        return new_dist, new_LB, new_dot_product
 
    def ComputeSubMP(self,idx:int,next_idx:int)->tuple:
        """ Compute the SubMatrixProfile from the profile of len l to the profile of len l+k
            Args: 
                idx(int): window length index
                next_idx: next window length index
            Returns:
                bBestM(Bool): indicate if the subMP is sufficient to obtain the whole MatrixProfile
                SubMP(np.ndarray): subMatrixProfile 
                IP(np.ndarray): SubIndexProfile
        """
        SubMP=np.zeros(self.nDP)
        SubIP = np.zeros(self.nDP).astype(int)
        minDistAbs, minLBAbs = np.inf, np.inf
        nonValidDP=[]
        for j in range(self.nDP):
            minDist = np.inf
            DP=self.listDP[j]
            #the maximum LB is at the last position
            maxLB=DP.LB[-1]
            for e in range(self.p):
                i=DP.idxs[e]
                dot_product=DP.dot_product[e]
                LB=DP.LB[e]
                e_dist, e_LB, e_dot_product = self.updateDistAndLB(idx,next_idx,i,j,dot_product,LB)
                DP.distance[e], DP.LB[e], DP.dot_product[e] = e_dist, e_LB, e_dot_product
                minDist=min(minDist,e_dist)
                if minDist==e_dist:
                    ind=i
            #we update DP
            self.listDP[j]=DP
            if minDist<maxLB:
                minDistAbs=min(minDistAbs,minDist)
                SubMP[j]=minDist
                SubIP[j]=ind
            else:
                minLBAbs=min(minLBAbs,maxLB)
                SubMP[j]=np.inf
                nonValidDP.append([j,maxLB])
                
                
        bBestM=minDistAbs<minLBAbs
        
        if (not bBestM) and len(nonValidDP)<(self.nDP*np.log(self.p)/np.log(self.nDP)):
            for ind,maxLB in nonValidDP:
                if maxLB<minDistAbs:
                    line=self.distance_[next_idx].first_line(ind)
                    SubIP[ind]=np.argmin(line)
                    SubMP[ind]=np.min(line)
                    
                    non_overlap_mask = np.arange(max(0,i-self.wlens_[next_idx]+1), min(self.nDP,i+self.wlens_[next_idx]))

                    LB=self.CompLB(next_idx,next_idx+1,ind)
                    
                    idx_sort=np.argsort(LB)[:self.p]
                    trunc_dist=line[idx_sort]
                    trunc_LB=LB[idx_sort]
                    trunc_dot_prod=self.distance_[next_idx].dot_product_[idx_sort]
                    #instead of creating a new DP maybe updating it ?
                    
                    listDP[ind]=(DP(idx_sort, trunc_dist, trunc_LB, trunc_dot_prod))
                    
            bBestM=True

        return bBestM,SubMP, IP

        
    def Valmod(self,signal:np.ndarray)->np.ndarray:
        """
        Compute the Variable length matrix profile (VALMP)
        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)

        """
        self.signal_=signal
        self.distance_=[]
        self.wlens_ = np.arange(self.min_wlen,self.max_wlen,self.step)
        self.nDP=self.signal_.shape[0]-self.min_wlen+1
        for idx,wlen in enumerate(self.wlens_):
            #à changer si on veut pouvoir changer de distance
            self.distance_.append(getattr(distance,self.distance_name)(self.wlens_[idx],**self.distance_params))
            self.distance_[idx].fit(self.signal_)

        MP, IP, self.listDP = self.ComputeMatrixProfile(0)
        self.VALMP = VALMP(self.nDP)
        self.VALMP.updateVALMP(MP,IP,self.wlens_[0])

        for idx,wlen in enumerate(self.wlens_[1:],start=1):
            self.nDP= self.signal_.shape[0]-wlen+1
            self.VALMP.nDP=self.nDP
            bBestM, SubMP, SubIP = self.ComputeSubMP(idx-1,idx)
            print(wlen)
            print(bBestM)
            if bBestM:
                self.VALMP.updateVALMP(SubMP,SubIP,wlen)
            else:
                MP, IP, self.listDP = self.ComputeMatrixProfile(idx)
                self.VALMP.updateVALMP(MP, IP, wlen)


class VALMP(object):
    def __init__(self, nDP:int) -> None:
        """Initialization
            Args: nDP = initial number of subsequences
        """
        self.nDP = nDP
        self.distances = np.inf * np.ones(self.nDP)
        self.normDistances = np.inf * np.ones(self.nDP)
        self.lengths = np.zeros(self.nDP)
        self.indices = np.zeros(self.nDP)

    def updateVALMP(self, MPnew:np.ndarray, IP:np.ndarray, wlen:int) -> None:
        """Update of VALMP
            Args: 
                MPnew(np.ndarray): matrix profile for the current length
                IP(np.ndarray): index profile for the current length
                wlen(int): current length
                """
        lNormDist = MPnew / np.sqrt(wlen)
        for i in range(self.nDP):
            if lNormDist[i] < self.normDistances[i]:
                self.distances[i]=MPnew[i]
                self.normDistances[i]=lNormDist[i]
                self.lengths[i] = wlen
                self.indices[i] = IP[i]
    
class DP(object):
    
    def __init__(self, idxs:np.ndarray, distance:np.ndarray, LB:np.ndarray, dot_product:np.ndarray )-> None:

        self.idxs = idxs
        self.distance = distance
        self.LB = LB
        self.dot_product =dot_product
        
    
