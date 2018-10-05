/*
  some very easy 例题 
  luogu P3367  【模板】并查集 
  luogu P1551   亲戚
  luogu P3958   奶酪
  luogu P1525   关押罪犯
  luogu P1196   银河英雄传说  (%%%达哥）
  luogu P2024   食物链
   
*/
#include<cstdio>
#include<iostream>
using namespace std;
int fa[maxn]; //存储每个节点的根节点(父亲)
//初始化  ==> for(int i=1;i<=maxn;i++) fa[i]=i; 即初始时自己的根节点均为自己 

int find(int v) //查询根节点 
{
	if(v==fa[v]) return v; //如果根节点为自己直接返回 
	else
	{
		fa[v]=find(fa[v]) //递归地向上找出根节点并把当前
  			             //节点的父亲节点标记为根节点 
        return fa[v];
	}
} /*
  	事实上，我们并不关心一个节点的父亲节点是谁，只需要关心它是否在一个集合中
	==> 路径压缩优化   
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

//以上只是不同写法0.0

void merge(int x,int y) //合并操作
{
	int t1=find(x) ,t2=find(y); //找到需要合并的两个节点的根节点 
	if(t1!=t2) // 若不在一个集合中 
			   fa[t2]=t1;  //合并
	return ;   
}
/*
  	关于某种名为启发式合并（按秩合并）的正统优化
	  ==> 钟(c)长(h)者(l)也不会 
	  于是若有被卡合并操作的可能(NOIP一般不会有）
	  ==> rand()合并大法好 
*/

void merge(int x,int y) 
{
	int t1=find(x),t2=find(y);
	if (t1!=t2) 
	{
		if (rand()%2) fa[t1]=t2; //玄学防卡 
		else fa[t2]=t1;
	}
}


int main(void)
{
	for(int i=1;i<=maxn;i++) fa[i]=i;
	char a[233];
	for(int i=1;i<=Q;i++) //Q表示Q次询问
	{
		cin>>a+1;
		if(a+1=="chaxun") 
		{
			int x,y; //读入需要查询的节点 
			scanf("%d%d",&x,&y);
			if(find(x)==find(y)) puts("in the same jihe");  
		}
		if(a+1=="hebing")
		{
			int x,y; //读入需要合并的节点 
			scanf("%d%d",&x,&y);
			merge(x,y);
		}
	} 
	return 0;
}
