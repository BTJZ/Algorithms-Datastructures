/*
  some very easy ���� 
  luogu P3367  ��ģ�塿���鼯 
  luogu P1551   ����
  luogu P3958   ����
  luogu P1525   ��Ѻ�ﷸ
  luogu P1196   ����Ӣ�۴�˵  (%%%��磩
  luogu P2024   ʳ����
   
*/
#include<cstdio>
#include<iostream>
using namespace std;
int fa[maxn]; //�洢ÿ���ڵ�ĸ��ڵ�(����)
//��ʼ��  ==> for(int i=1;i<=maxn;i++) fa[i]=i; ����ʼʱ�Լ��ĸ��ڵ��Ϊ�Լ� 

int find(int v) //��ѯ���ڵ� 
{
	if(v==fa[v]) return v; //������ڵ�Ϊ�Լ�ֱ�ӷ��� 
	else
	{
		fa[v]=find(fa[v]) //�ݹ�������ҳ����ڵ㲢�ѵ�ǰ
  			             //�ڵ�ĸ��׽ڵ���Ϊ���ڵ� 
        return fa[v];
	}
} /*
  	��ʵ�ϣ����ǲ�������һ���ڵ�ĸ��׽ڵ���˭��ֻ��Ҫ�������Ƿ���һ��������
	==> ·��ѹ���Ż�   
*/

int find(int v)
{
	if(v==fa[v]) return v;
	return fa[v]=find(fa[v]);
}

int find(int v)
{
	return v==fa[v]? v:fa[v]=find(fa[v]);
}

//����ֻ�ǲ�ͬд��0.0

void merge(int x,int y) //�ϲ�����
{
	int t1=find(x) ,t2=find(y); //�ҵ���Ҫ�ϲ��������ڵ�ĸ��ڵ� 
	if(t1!=t2) // ������һ�������� 
			   fa[t2]=t1;  //�ϲ�
	return ;   
}
/*
  	����ĳ����Ϊ����ʽ�ϲ������Ⱥϲ�������ͳ�Ż�
	  ==> ��(c)��(h)��(l)Ҳ���� 
	  �������б����ϲ������Ŀ���(NOIPһ�㲻���У�
	  ==> rand()�ϲ��󷨺� 
*/

void merge(int x,int y) 
{
	int t1=find(x),t2=find(y);
	if (t1!=t2) 
	{
		if (rand()%2) fa[t1]=t2; //��ѧ���� 
		else fa[t2]=t1;
	}
}


int main(void)
{
	for(int i=1;i<=maxn;i++) fa[i]=i;
	char a[233];
	for(int i=1;i<=Q;i++) //Q��ʾQ��ѯ��
	{
		cin>>a+1;
		if(a+1=="chaxun") 
		{
			int x,y; //������Ҫ��ѯ�Ľڵ� 
			scanf("%d%d",&x,&y);
			if(find(x)==find(y)) puts("in the same jihe");  
		}
		if(a+1=="hebing")
		{
			int x,y; //������Ҫ�ϲ��Ľڵ� 
			scanf("%d%d",&x,&y);
			merge(x,y);
		}
	} 
	return 0;
}
