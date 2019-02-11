##NN


##Perceptron
a<-0.2  
w<-rep(0,3)  
iris1<-t(as.matrix(iris[,3:4]))  
d<-c(rep(0,50),rep(1,100))  
e<-rep(0,150)  
p<-rbind(rep(1,150),iris1)  
max<-100000  
eps<-rep(0,100000)  
i<-0  
repeat{  
  v<-w%*%p;  
  y<-ifelse(sign(v)>=0,1,0);  
  e<-d-y;  
  eps[i+1]<-sum(abs(e))/length(e)  
  if(eps[i+1]<0.01){  
    print("finish:");  
    print(w);  
    break;  
  }  
  w<-w+a*(d-y)%*%t(p);  
  i<-i+1;  
  if(i>max){  
    print("max loop");  
    print(eps[i])  
    print(y);  
    break;  
  }  
}  
#plot
plot(y~x1,xlim=c(0,3),ylim=c(0,8),  
     data=mydata)  

x<-seq(0,3,0.01)  
y<-x*(-w[3]/w[2])-w[1]/w[2]  
lines(x,y,col=4)  
#plot average error 
plot(1:i,eps[1:i],type="o")  


##Linear Neural Network
p<-rbind(rep(1,150),iris1)  
d<-c(rep(0,50),rep(1,100))  
w<-rep(0,3)  
a<-1/max(eigen(t(p)%*%p)$values)  
max<-1000  
e<-rep(0,150)  
eps<-rep(0,1000)  
i<-0  
for(i in 1:max){  
  v<-w%*%p;  
  y<-v;  
  e<-d-y;  
  eps[i+1]<-sum(e^2)/length(e)  
  w<-w+a*(d-y)%*%t(p);  
  if(i==max)  
    print(w)  
}  

##BP Neural Network
 
set.seed(1992)  
n<-length(mydata)  
samp<-sample(1:n,n/5)  
traind<-mydata[-samp,c(1,2)]  
train1<-mydata[-samp,3]  
testd<-mydata[samp,c(1,2)]  
test1<-mydata[samp,3]  

set.seed(12)  
ntrainnum<-100
nsampdim<-2  

net.nin<-2  
net.nhidden<-3  
net.nout<-1  
w<-2*matrix(runif(net.nhidden*net.nin)-0.5,net.nhidden,net.nin)  
b<-2*(runif(net.nhidden)-0.5)  
net.w1<-cbind(w,b)  
W<-2*matrix(runif(net.nhidden*net.nout)-0.5,net.nout,net.nhidden)  
B<-2*(runif(net.nout)-0.5)  
net.w2<-cbind(W,B)  

traind_s<-traind  
traind_s[,1]<-traind[,1]-mean(traind[,1])  
traind_s[,2]<-traind[,2]-mean(traind[,2])  
traind_s[,1]<-traind_s[,1]/sd(traind_s[,1])  
traind_s[,2]<-traind_s[,2]/sd(traind_s[,2])  

sampinex<-rbind(t(traind_s),rep(1,ntrainnum))  
expectedout<-train1  

eps<-0.01  
a<-0.3  
mc<-0.8  
maxiter<-2000  
iter<-0  

errrec<-rep(0,maxiter)  
outrec<-matrix(rep(0,ntrainnum*maxiter),ntrainnum,maxiter)  

##Sigmoid Function
sigmoid<-function(x){  
  y<-1/(1+exp(-x))  
  return(y)  
}  

for(i in 1:maxiter){  
  hid_input<-net.w1%*%sampinex;  
  hid_out<-sigmoid(hid_input);  
  out_input1<-rbind(hid_out,rep(1,ntrainnum));  
  out_input2<-net.w2%*%out_input1;  
  out_out<-sigmoid(out_input2);  
  outrec[,i]<-t(out_out);  
  err<-expectedout-out_out;  
  sse<-sum(err^2);  
  errrec[i]<-sse;  
  iter<-iter+1;  
  if(sse<=eps)  
    break  
  
  Delta<-err*sigmoid(out_out)*(1-sigmoid(out_out))  
  delta<-(matrix(net.w2[,1:(length(net.w2[1,])-1)]))%*%Delta*sigmoid(hid_out)*(1-sigmoid(hid_out));  
  
  dWex<-Delta%*%t(out_input1)  
  dwex<-delta%*%t(sampinex)  
  
  if(i==1){  
    net.w2<-net.w2+a*dWex;  
    net.w1<-net.w1+a*dwex;  
  }  
  else{  
    net.w2<-net.w2+(1-mc)*a*dWex+mc*dWexold;  
    net.w1<-net.w1+(1-mc)*a*dwex+mc*dwexold;  
  }  
  
  dWexold<-dWex;  
  dwexold<-dwex;  
}  


testd_s<-testd  
testd_s[,1]<-testd[,1]-mean(testd[,1])  
testd_s[,2]<-testd[,2]-mean(testd[,2])  
testd_s[,1]<-testd_s[,1]/sd(testd_s[,1])  
testd_s[,2]<-testd_s[,2]/sd(testd_s[,2])  

inex<-rbind(t(testd_s),rep(1,150-ntrainnum))  
hid_input<-net.w1%*%inex  
hid_out<-sigmoid(hid_input)  
out_input1<-rbind(hid_out,rep(1,150-ntrainnum))  
out_input2<-net.w2%*%out_input1  
out_out<-sigmoid(out_input2)  
out_out1<-out_out  

out_out1[out_out<0.5]<-0  
out_out1[out_out>=0.5]<-1  

rate<-sum(out_out1==test1)/length(test1)  