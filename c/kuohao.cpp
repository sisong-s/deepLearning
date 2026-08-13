#include <iostream>
#include <string>
#include <vector>
#include <climits>
using namespace std;

// 判断两个括号是否成对匹配
bool isMatch(char l, char r)
{
    return (l == '(' && r == ')') || (l == '[' && r == ']');
}

// 获取单个括号配对字符串
string getPair(char c)
{
    if (c == '(' || c == ')')
        return "()";
    else
        return "[]";
}

string minBracketComplete(string s)
{
    int n = s.size();
    if (n == 0)
        return "";

    vector<vector<int>> dp(n, vector<int>(n, 0));
    vector<vector<string>> str(n, vector<string>(n));

    // 长度为1的子串初始化
    for (int i = 0; i < n; ++i)
    {
        dp[i][i] = 1;
        str[i][i] = getPair(s[i]);
    }

    // 枚举区间长度 len 从2到n
    for (int len = 2; len <= n; ++len)
    {
        for (int i = 0; i + len - 1 < n; ++i)
        {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;

            // 首尾字符匹配，直接向内收缩
            if (isMatch(s[i], s[j]))
            {
                dp[i][j] = dp[i + 1][j - 1];
                str[i][j] = s[i] + str[i + 1][j - 1] + s[j];
            }

            // 枚举分割点k，左右区间合并取最优解
            for (int k = i; k < j; ++k)
            {
                int cost = dp[i][k] + dp[k + 1][j];
                if (cost < dp[i][j])
                {
                    dp[i][j] = cost;
                    str[i][j] = str[i][k] + str[k + 1][j];
                }
            }
        }
    }
    return str[0][n - 1];
}

int main()
{
    vector<string> test = {"([", "])", "([)]", "([[]", ""};
    for (auto &s : test)
    {
        string res = minBracketComplete(s);
        cout << "原串: " << s << "\t补全: " << res
             << "\t新增括号数: " << res.size() - s.size() << endl;
    }
    return 0;
}
