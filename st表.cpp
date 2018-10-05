//syc 区间RMQ最大值
//洛谷 P3865
//提示 数据过大不要用cin
//cout 也不能用
#include<cstdio>
#include<iostream>
#include<cstdio>
#include<cmath>
using namespace std;

const int maxn = 100000+5; 

int n,m;

int bit[25],st[20][maxn],lg[maxn];

void in(){
	cin>>n>>m;
	for(int i = 1 ; i <= n ; i++) scanf("%d",&st[0][i]);
}

void make(){
	bit[0] = 1; for(int i = 1 ; i <= 24 ; i++) bit[i] = bit[i-1] * 2;
	for(int i = 2 ; i <= n ; i++) lg[i] = lg[i>>1] + 1;
	for(int i = 1 ; i <= lg[n] ; i++)
		for(int j = 1 ; j <= n ;j++)
			st[i][j] = max(st[i-1][j],st[i-1][j+bit[i-1]]);
}

void find(){
	int l,r;
	scanf("%d%d",&l,&r);
	int k = lg[r-l+1];
	printf("%d\n",max(st[k][l],st[k][r-bit[k]+1]));
}

int main(){
	in();
	make();
	for(int i = 1 ; i <= m ; i++)	
	find();
	return 0;
}
