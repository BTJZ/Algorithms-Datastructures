//syc 
//P1082 同余方程
//求逆元
#include<iostream>
#include<cstdio>
#include<cmath>
using namespace std;

int x,y,c;
//c 为最大公约数
void ex_gcd(int a,int b,int &x,int &y){
	if(b==0){
		c = a;
		x = 1;
		y = 0;
		return ;
	}
	ex_gcd(b,a%b,y,x);
	y -= x*(a/b);
}

int main(){
	int a,b;
	cin>>a>>b;
	ex_gcd(a,b,x,y);
	cout<<(x+b)%b;
	return 0;
}
