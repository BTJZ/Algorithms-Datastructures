#include<iostream>
#include<cstdio>
using namespace std;

const int maxn = ;

int map[maxn][maxn];

int n,m;

void floyd(){
	for(int k = 1 ; k <= n ; k++)
		for(int i = 1 ; i <= n ; i++)
			for(int j = 1 ; j <= n ; j++)
				if(map[i][k]+map[k][j]<map[i][j])
				map[i][j] = map[i][k] + map[k][j];
}

int main(){
	in();
	floyd();
	out();
	return 0;
}
