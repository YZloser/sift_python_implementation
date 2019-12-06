import cv2
import numpy as np
import math
import random
SIFT_INIT_SIGMA=0.5
SIFT_SIGMA=1.6
SIFT_INTVLS=3
SIFT_CONTR_THR=0.04
SIFT_CURV_THR=10
SIFT_IMG_BORDER=5
SIFT_MAX_INTERP_STEPS=5
SIFT_ORI_HIST_BINS=36
SIFT_ORI_SIG_FCTR=1.5
SIFT_ORI_RADIUS=3.0*SIFT_ORI_SIG_FCTR
SIFT_ORI_SMOOTH_PASSES=2
SIFT_ORI_PEAK_RATIO=0.8
SIFT_DESCR_HIST_BINS=8
SIFT_DESCR_WIDTH=4
SIFT_DESCR_SCL_FCTR=3.0
SIFT_DESCR_MAG_THR=0.2
SIFT_INT_DESCR_FCTR=512.0
SIFT_MATCH_THR=0.2

    
class Feature(object):
    def __init__(self,x,y,i,j,r,c,sj,scl=0,scl_octv=0):
        self.x=x
        self.y=y
        self.i=i
        self.j=j
        self.r=r
        self.c=c
        self.sj=sj
        self.scl=scl
        self.scl_octv=scl_octv
        self.descr=[]

def show(img):
    img=img.astype(np.uint8)
    cv2.imshow("Image",img)
    cv2.waitKey(0)

def convert_to_grayfloat(img):
    if img.ndim==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)/255.0

def create_init_img(img,sigma):
    img=cv2.resize(img,(img.shape[1]*2,img.shape[0]*2),cv2.INTER_CUBIC)
    sigma=math.sqrt(sigma**2-(SIFT_INIT_SIGMA**2)*4)
    img=cv2.GaussianBlur(img,(0,0),sigma)
    return img

def down_sample(img):
    return cv2.resize(img,(img.shape[1]/2,img.shape[0]/2),cv2.INTER_NEAREST)

def build_gauss_pyr(img,octaves,intvls,sigma):
    k=2.0**(1.0/intvls)
    sig=[0]*(intvls+3)
    sig[0]=sigma
    sig[1]=sigma*math.sqrt(k*k-1)
    gauss_pyr=[[0 for i in range(intvls+3)] for j in range(octaves)]
    for i in range(2,intvls+3):
        sig[i]=sig[i-1]*k
    for i in range(octaves):
        for j in range(intvls+3):
            if i==0 and j==0:
                gauss_pyr[i][j]=img
            elif j==0:
                gauss_pyr[i][j]=down_sample(gauss_pyr[i-1][intvls])
            else:
                gauss_pyr[i][j]=cv2.GaussianBlur(gauss_pyr[i][j-1],(0,0),sig[j])
    return gauss_pyr

def build_dog_pyr(gauss_pyr,octaves,intvls):
    dog_pyr=[[0 for i in range(intvls+2)] for j in range(octaves)]
    for i in range(octaves):
        for j in range(intvls+2):
            dog_pyr[i][j]=gauss_pyr[i][j+1]-gauss_pyr[i][j]
    return dog_pyr

def is_extrema(dog_pyr,i,j,r,c):
    val=dog_pyr[i][j][r][c]
    for x in range(-1,2):
        for y in range(-1,2):
            for z in range(-1,2):
                if val>=0 and val<dog_pyr[i][j+x][r+y][c+z]:
                    return False
                elif val<0 and val>dog_pyr[i][j+x][r+y][c+z]:
                    return False
    return True

def deriv_3D(dog_pyr,i,j,r,c):
    dx=(dog_pyr[i][j][r][c+1]-dog_pyr[i][j][r][c-1])/2.0
    dy=(dog_pyr[i][j][r+1][c]-dog_pyr[i][j][r-1][c])/2.0
    ds=(dog_pyr[i][j+1][r][c]-dog_pyr[i][j-1][r][c])/2.0
    return np.array([[dx],[dy],[ds]])

def hessian_3D(dog_pyr,i,j,r,c):
    v=dog_pyr[i][j][r][c]
    dxx=dog_pyr[i][j][r][c+1]+dog_pyr[i][j][r][c-1]-2*v
    dyy=dog_pyr[i][j][r+1][c]+dog_pyr[i][j][r-1][c]-2*v
    dss=dog_pyr[i][j+1][r][c]+dog_pyr[i][j-1][r][c]-2*v
    dxy=(dog_pyr[i][j][r+1][c+1]-dog_pyr[i][j][r+1][c-1]-dog_pyr[i][j][r-1][c+1]+dog_pyr[i][j][r-1][c-1])/4.0
    dxs=(dog_pyr[i][j+1][r][c+1]-dog_pyr[i][j+1][r][c-1]-dog_pyr[i][j-1][r][c+1]+dog_pyr[i][j-1][r][c-1])/4.0
    dys=(dog_pyr[i][j+1][r+1][c]-dog_pyr[i][j+1][r-1][c]-dog_pyr[i][j-1][r+1][c]+dog_pyr[i][j-1][r-1][c])/4.0
    return np.array([[dxx,dxy,dxs],
                     [dxy,dyy,dys],
                     [dxs,dys,dss]])

def interp_step(dog_pyr,i,j,r,c):
    dD=deriv_3D(dog_pyr,i,j,r,c)
    H=hessian_3D(dog_pyr,i,j,r,c)
    H_inv=np.linalg.inv(H)
    return -H_inv.dot(dD)
    
def interp_contr(dog_pyr,i,j,r,c,t):
    dD=deriv_3D(dog_pyr,i,j,r,c).transpose()
    return dD.dot(t)[0][0]*0.5+dog_pyr[i][j][r][c]

def interp_extremum(dog_pyr,i,j,r,c,intvls,contr_thr):
    x=0
    t=0
    while x<SIFT_MAX_INTERP_STEPS:
        t=interp_step(dog_pyr,i,j,r,c)
        if abs(t[0][0])<0.5 and abs(t[1][0])<0.5 and abs(t[2][0])<0.5:
            break
        j+=int(round(t[2][0])+0.1)
        r+=int(round(t[1][0])+0.1)
        c+=int(round(t[0][0])+0.1)
        if j<1 or j>intvls or r<SIFT_IMG_BORDER or r>dog_pyr[i][0].shape[0]-SIFT_IMG_BORDER or c<SIFT_IMG_BORDER or c>dog_pyr[i][0].shape[1]-SIFT_IMG_BORDER:
            return None
        x+=1
    if x>=SIFT_MAX_INTERP_STEPS:
        return None
    contr=interp_contr(dog_pyr,i,j,r,c,t)
    if abs(contr)<float(contr_thr)/intvls:
        return None
    return Feature((c+t[0][0])*(2.0**i),(r+t[1][0])*(2.0**i),i,j,r,c,t[2][0])
    
def is_too_edge_like(dog_img,r,c,curv_thr):
    d=dog_img[r][c]
    dxx=dog_img[r][c+1]+dog_img[r][c-1]-2*d
    dyy=dog_img[r+1][c]+dog_img[r-1][c]-2*d
    dxy=(dog_img[r+1][c+1]-dog_img[r+1][c-1]-dog_img[r-1][c+1]+dog_img[r-1][c-1])/4.0
    tr=dxx+dyy
    det=dxx*dyy-dxy*dxy
    if det<=0:
        return True
    if tr*tr/det<(curv_thr+1.0)*(curv_thr+1.0)/curv_thr:
        return False
    return True
    
def find_extrema(dog_pyr,octaves,intvls,contr_thr,curv_thr):
    features=[]
    prelim_contr_thr=0.5*contr_thr/intvls
    for i in range(octaves):
        for j in range(1,intvls+1):
            for r in range(SIFT_IMG_BORDER,dog_pyr[i][0].shape[0]-SIFT_IMG_BORDER):
                for c in range(SIFT_IMG_BORDER,dog_pyr[i][0].shape[1]-SIFT_IMG_BORDER):
                    if abs(dog_pyr[i][j][r][c])>prelim_contr_thr:
                        if is_extrema(dog_pyr,i,j,r,c):
                            feat=interp_extremum(dog_pyr,i,j,r,c,intvls,contr_thr)
                            if feat:
                                if not is_too_edge_like(dog_pyr[i][j],feat.r,feat.c,curv_thr):
                                    features.append(feat)
    return features

def calc_feature_scales(features,sigma,intvls):
    for i in features:
        intvl=i.sj+i.j
        i.scl=sigma*(2.0**(i.i+intvl/intvls))
        i.scl_octv=sigma*(2.0**(intvl/intvls))

def adjust(features):
    for i in features:
        i.x/=2.0
        i.y/=2.0
        i.scl/=2.0
    
def calc_grad_mag_ori(img,r,c):
    if r>0 and r<img.shape[0]-1 and c>0 and c<img.shape[1]-1:
        dx=img[r][c+1]-img[r][c-1]
        dy=img[r-1][c]-img[r+1][c]
        return (math.sqrt(dx*dx+dy*dy),math.atan2(dy,dx))
    return None

def ori_hist(img,r,c,n,rad,sigma):
    exp_denom=2.0*sigma*sigma
    PI2=math.pi*2.0
    hist=[0]*n
    for i in range(-rad,rad+1):
        for j in range(-rad,rad+1):
            res=calc_grad_mag_ori(img,r,c)
            if res:
                mag=res[0]
                ori=res[1]
            w=math.exp(-(i*i+j*j)/exp_denom)
            bi=int(round(n*(ori+math.pi)/PI2)+0.1)
            bi=(bi if(bi<n) else 0)
            hist[bi]+=w*mag
    return hist

def smooth_ori_hist(hist,n):
    prev=hist[n-1]
    for i in range(n):
        tmp=hist[i]
        hist[i]=0.25*prev+0.5*hist[i]+0.25*hist[(i+1)%n]
        prev=tmp

def dominant_ori(hist,n):
    omax = hist[0]
    maxbin = 0
    for i in range(1,n):
        if hist[i]>omax:
            omax=hist[i]
            maxbin=i
    return omax

def interp_hist_peak(l,c,r):
    return 0.5*((l)-(r))/((l)-2.0*(c)+(r))

def add_good_ori_features(features,hist,n,mag_thr,feat):
    PI2=math.pi*2.0
    tmp_f=[]
    for i in range(n):
        l=(i-1)%n
        r=(i+1)%n
        if hist[i]>hist[l] and hist[i]>hist[r] and hist[i]>mag_thr:
            bi=i+interp_hist_peak(hist[l],hist[i],hist[r])
            bi%=n
            f=Feature(feat.x,feat.y,feat.i,feat.j,feat.r,feat.c,feat.sj,feat.scl,feat.scl_octv)
            f.ori=((PI2*bi)/n)-math.pi
            tmp_f.append(f)
    return tmp_f

def calc_feature_oris(features,gauss_pyr):
    real_features=[]
    for feat in features:
        hist=ori_hist(gauss_pyr[feat.i][feat.j],feat.r,feat.c,SIFT_ORI_HIST_BINS,int(round(SIFT_ORI_RADIUS*feat.scl_octv)+0.1),SIFT_ORI_SIG_FCTR*feat.scl_octv)
        for i in range(SIFT_ORI_SMOOTH_PASSES):
            smooth_ori_hist(hist,SIFT_ORI_HIST_BINS)
        omax=dominant_ori(hist,SIFT_ORI_HIST_BINS)
        real_features+=add_good_ori_features(features,hist,SIFT_ORI_HIST_BINS,omax*SIFT_ORI_PEAK_RATIO,feat)
    return real_features

def interp_hist_entry(hist,rbin,cbin,obin,mag,d,n):
    r0=int(math.floor(rbin)+0.1)
    c0=int(math.floor(cbin)+0.1)
    o0=int(math.floor(obin)+0.1)
    d_r=rbin-r0
    d_c=cbin-c0
    d_o=obin-o0
    for r in range(2):
        rb=r0+r
        if rb>=0 and rb<d:
            v_r=mag*(1.0-d_r if(r==0) else d_r)
            row=hist[rb]
            for c in range(2):
                cb=c0+c
                if cb>=0 and cb<d:
                    v_c=v_r*(1.0-d_c if(c==0) else d_c)
                    h=row[cb]
                    for o in range(2):
                        ob=(o0+o)%n
                        v_o=v_c*(1.0-d_o if(c==0) else d_o)
                        h[ob]+=v_o

def descr_hist(img,r,c,ori,scl,d,n):
    hist=[[[0 for i in range(n)] for j in range(d)] for k in range(d)]
    cos_t=math.cos(ori)
    sin_t=math.sin(ori)
    PI2=math.pi*2.0
    bins_per_rad=n/PI2
    exp_denom=d*d*0.5
    hist_width=SIFT_DESCR_SCL_FCTR*scl
    radius=hist_width*math.sqrt(2)*(d+1.0)*0.5+0.5
    i=int(-radius)
    while i<=radius:
        j=int(-radius)
        while j<=radius:
            c_rot=(j*cos_t-i*sin_t)/hist_width
            r_rot=(j*sin_t+i*cos_t)/hist_width
            rbin=r_rot+d/2.0-0.5
            cbin=c_rot+d/2.0-0.5
            if rbin>-1.0 and rbin<d and cbin>-1.0 and cbin<d:
                grad=calc_grad_mag_ori(img,r+i,c+j)
                if grad:
                    grad_mag=grad[0]
                    grad_ori=grad[1]
                    grad_ori-=ori
                    while grad_ori<0.0:
                        grad_ori+=PI2
                    while grad_ori >= PI2:
                        grad_ori-=PI2
                    obin=grad_ori*bins_per_rad
                    w=math.exp(-(c_rot*c_rot+r_rot*r_rot)/exp_denom )
                    interp_hist_entry(hist,rbin,cbin,obin,grad_mag*w,d,n)
            j+=1
        i+=1
    return hist

def normalize_descr(feat):
    len_sq=0.0
    d=len(feat.descr)
    for i in range(d):
        cur=feat.descr[i]
        len_sq+=cur*cur
    len_inv=1.0/math.sqrt(len_sq)
    for i in range(d):
        feat.descr[i]*=len_inv

def hist_to_descr(hist,d,n,feat):
    for r in range(d):
        for c in range(d):
            for o in range(n):
                feat.descr.append(hist[r][c][o])
    normalize_descr(feat)
    for i in range(len(feat.descr)):
        if feat.descr[i]>SIFT_DESCR_MAG_THR:
            feat.descr[i]=SIFT_DESCR_MAG_THR
    normalize_descr(feat)
    for i in range(len(feat.descr)):
        int_val=int(SIFT_INT_DESCR_FCTR*feat.descr[i])
        feat.descr[i]=min(255,int_val)

def compute_descriptors(features,gauss_pyr,d,n):
    for feat in features:
        hist=descr_hist(gauss_pyr[feat.i][feat.j],feat.r,feat.c,feat.ori,feat.scl_octv,d,n)
        hist_to_descr(hist,d,n,feat)

def sift_feature(img):
    _=img
    sigma=SIFT_SIGMA
    intvls=SIFT_INTVLS
    contr_thr=SIFT_CONTR_THR
    curv_thr=SIFT_CURV_THR
    descr_width=SIFT_DESCR_WIDTH
    descr_hist_bins=SIFT_DESCR_HIST_BINS
    img=convert_to_grayfloat(img)
    init_img=create_init_img(img,sigma)
    height,width=init_img.shape
    octaves=int(math.log(min(height,width),2))-2
    gauss_pyr=build_gauss_pyr(init_img,octaves,intvls,sigma)
    dog_pyr=build_dog_pyr(gauss_pyr,octaves,intvls) 
    features=find_extrema(dog_pyr,octaves,intvls,contr_thr,curv_thr)
    calc_feature_scales(features,sigma,intvls)
    adjust(features)
    real_features=calc_feature_oris(features,gauss_pyr)
    compute_descriptors(real_features,gauss_pyr,descr_width,descr_hist_bins)
    return real_features

def distance(l1,l2):
    ans=0
    for i in range(len(l1.descr)):
        ans+=(l1.descr[i]-l2.descr[i])**2.0
    return math.sqrt(ans)

def brute_force_match(orgin,compare,alpha):
    match=[]
    if len(orgin)<2:
        return None
    for feat in compare:
        d1=distance(feat,orgin[0])
        d2=distance(feat,orgin[1])
        fstd=min(d1,d2)
        scdd=max(d1,d2)
        fstf=orgin[0] if (d1<d2) else orgin[1]
        for i in orgin:
            d=distance(i,feat)
            if d<fstd:
                scdd=fstd
                fstd=d
                fstf=i
            elif d<scdd:
                scdd=d
        if fstd/scdd<alpha:
            match.append((fstf,feat))
    return  match

def draw_pic(img1,img2,match):
    new_shape=(max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3)
    new_pic=np.zeros(new_shape)
    new_pic[0:img1.shape[0],0:img1.shape[1]]=img1
    new_pic[0:img2.shape[0],img1.shape[1]:]=img2
    for i in match:
        cv2.line(new_pic,(int(round(i[0].x+0.1)),int(round(i[0].y)+0.1)),(int(round(i[1].x)+0.1)+img1.shape[1],int(round(i[1].y)+0.1)),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),2)
    show(new_pic)
        
if __name__=='__main__':
    img1=cv2.imread("2.jpg")
    img2=cv2.imread("3.jpg")
    features1=sift_feature(img1)
    features2=sift_feature(img2)
    match=brute_force_match(features1,features2,SIFT_MATCH_THR)
    draw_pic(img1,img2,match)
    
    
