#include<stdio.h>

#include<stdlib.h>

#include<time.h>

#include<pthread.h>



pthread_t thid[1000][1000];//n*n线程线程id

pthread_t tid[2];//双线程线程id

int pass[1000][1000][2];

int n;//矩阵阶数

int a[1000][1000],b[1000][1000];

int result[1000][1000];

int result1[1000][1000];


void* Thread1(void *arg)
{

int i,j,m;
for(i=0;(i<n);i++)

for(j=0;j<n;j++)

for(m=0;m<n;m++)

{

if(i%2==0)

result[i][j]+=a[i][m]*b[m][j];

}

}
void* Thread2(void *arg)

{

int i,j,m;
for(i=0;(i<n);i++)

for(j=0;j<n;j++)

for(m=0;m<n;m++)

{

if(i%2!=0)

result[i][j]+=a[i][m]*b[m][j];

}

}

//---------------多线程操作函数----------------------
void* Thread(void *arg)

{
int*  p = (int*) arg;
int row=p[0];

int col=p[1];

int res=0;

int l;

for(l=0;l<n;l++)
res+=a[row][l]*b[l][col];

result[row][col]=res;

}

//---------------------主函数------------------------
int main()

{

int t;

for(t=1;t<200;t++)

{

int i,j,m;//循环变量

double start1,finish1;//时间变量

double start2,finish2;

double start3,finish3;

int r;

n=t;

//生成随机数组、输出矩阵a，b

srand((unsigned) time(NULL));//生成时间种子

for(i=0;i<n;i++)

for(j=0;j<n;j++)

{

a[i][j]=rand()%11;

b[i][j]=rand()%11;
}

//--------------------------n*n多线程运算-------------------

start1=clock();//获得开始时间

for(i=0;i<n;i++)

for(j=0;j<n;j++)
{

pass[i][j][0]=i;

pass[i][j][1]=j;
r=pthread_create(&thid[i][j],NULL,Thread,pass[i][j]);
pthread_join(thid[i][j],NULL);//!!!!!!!!!!!!
}

finish1=clock();//获得结束时间

//----------------------双线程运算---------------------------

start3=clock();

pthread_create(&tid[0],NULL,Thread1,NULL);

pthread_join(tid[0],NULL);//!!!!!!!!!!!!

pthread_create(&tid[1],NULL,Thread2,NULL);

pthread_join(tid[1],NULL);//!!!!!!!!!!!!

finish3=clock();

//---------------------单线程运算-----------------------------

start2=clock();

for(i=0;i<n;i++)

for(j=0;j<n;j++)
for(m=0;m<n;m++)

result[i][j]+=a[i][m]*b[m][j];

finish2=clock();

printf("矩阵阶数: %3d |  Single: %10g  | Doube: %10g | n*n: %10g\n",t,(finish2-start2),(finish3-start3),(finish1-start1));

}

return 0;
}

