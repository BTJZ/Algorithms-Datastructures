//归并排序（分治思想）
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cmath>
using namespace std;
int n,a[1100000],c[1100000];//a为待排序数组，c为中间数组，不要忘记开两倍
long long ans;              //求逆序对数
void gb(int x,int y)        //此处为归并排序
{
    if(x==y)  return;       //当区间减为1时，返回
    int mid=(x+y)/2;        //将区间二分
    gb(x,mid);              //左右分别进行归并排序(分治)
    gb(mid+1,y); 
    int i=x,j=mid+1,k=x;    //此时左右区间分别有序，i，j为左右区间首，k为此段区间首
    while(i<=mid&&j<=y)     //当左右区间都未放进中间数组时
    {
        if(a[i]<=a[j])          
        {c[k]=a[i];k++;i++;}     //谁小谁先放进中间数组，指针后移一位
        else
        {c[k]=a[j];k++;j++;
        ans+=mid-i+1;} 			  //此处为用归并排序求逆序对数，详见逆序对定义，注意：“天对天多一天”	   
    } 	 
    while(i<=mid)                
    {c[k]=a[i];k++;i++;}         //如果左或右仍有未放入的，一定大于之前放入的
    while(j<=y)                  //直接放进中间数组
    {c[k]=a[j];k++;j++;}
    for(i=x;i<=y;i++)
    a[i]=c[i];                   //中间数组压入原数组
}
int main()
{
 	int i,j;
    scanf("%d",&n);
    for(i=1;i<=n;i++)  scanf("%d",&a[i]);       //读入
    gb(1,n);                                    //开始归并排序
    cout<<ans;
    return 0;
}


/*例题：
P1908 逆序对
P1966 火柴排队
P1309 瑞士轮*/
