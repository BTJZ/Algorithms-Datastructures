//�鲢���򣨷���˼�룩
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cmath>
using namespace std;
int n,a[1100000],c[1100000];//aΪ���������飬cΪ�м����飬��Ҫ���ǿ�����
long long ans;              //���������
void gb(int x,int y)        //�˴�Ϊ�鲢����
{
    if(x==y)  return;       //�������Ϊ1ʱ������
    int mid=(x+y)/2;        //���������
    gb(x,mid);              //���ҷֱ���й鲢����(����)
    gb(mid+1,y); 
    int i=x,j=mid+1,k=x;    //��ʱ��������ֱ�����i��jΪ���������ף�kΪ�˶�������
    while(i<=mid&&j<=y)     //���������䶼δ�Ž��м�����ʱ
    {
        if(a[i]<=a[j])          
        {c[k]=a[i];k++;i++;}     //˭С˭�ȷŽ��м����飬ָ�����һλ
        else
        {c[k]=a[j];k++;j++;
        ans+=mid-i+1;} 			  //�˴�Ϊ�ù鲢����������������������Զ��壬ע�⣺��������һ�족	   
    } 	 
    while(i<=mid)                
    {c[k]=a[i];k++;i++;}         //������������δ����ģ�һ������֮ǰ�����
    while(j<=y)                  //ֱ�ӷŽ��м�����
    {c[k]=a[j];k++;j++;}
    for(i=x;i<=y;i++)
    a[i]=c[i];                   //�м�����ѹ��ԭ����
}
int main()
{
 	int i,j;
    scanf("%d",&n);
    for(i=1;i<=n;i++)  scanf("%d",&a[i]);       //����
    gb(1,n);                                    //��ʼ�鲢����
    cout<<ans;
    return 0;
}


/*���⣺
P1908 �����
P1966 ����Ŷ�
P1309 ��ʿ��*/
