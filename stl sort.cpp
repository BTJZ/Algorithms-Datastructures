#include <algorithm>
using namespace std;//�������sort��ôһ����������������
#define maxn 10000
int a[maxn+5];//����a���飬��СΪmaxn+5
struct jgt
{
    int x,y;
}px[maxn+5];//����ṹ��
bool cmp(jgt p,jgt q)//����Ҫ�����������ĳ�ṹ��
{
    return p.x>q.x;//��x����
}
bool cmp(jgt p,jgt q)
{
    if(p.x>q.x)  return 1;
    else if(p.x<q.x)  return 0;
    else
    {
        if(p.y>q.y)  return 1;
        else  return 0;
    }//��xΪ��һ���ȣ���yΪ�ڶ���������
}
bool cmp(jgt p,jgt q)
{
    return p.x==q.x?p.y>q.y:p.x>q.x;
}//����д��Ϊ��Ŀ�����������һ�ֵȼ�
.
.
.

sort(a+1,a+maxn+1);//sort��������������һ��Ϊ�����������ʼ��ַ��
                   //�ڶ����ǽ����ĵ�ַ�����һλҪ����ĵ�ַ����һ��ַ����
                   //����������������ķ����������ǴӴ�СҲ���Ǵ�С���󣬻����Բ�д��������������ʱĬ�ϵ����򷽷��Ǵ�С��������
sort(px+1,px+maxn+1,cmp);//cmp��Ϊ����ʽ



//������������������������

