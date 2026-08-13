#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    string digui(string s, int r, int w)
    {
        string ans = "";
        for (int i = r; i <= w; i++)
        {
            if (s[i] == '#')
            {
                if (s[i + 1] == '{')
                {
                    int j = i + 2;
                    while (j <= w && s[j] != '}')
                    {
                        j++;
                    }
                    string pattern = s.substr(i + 2, j - i - 2);
                    string sub = digui(s, j + 1, w);
                    if (pattern == "upper")
                    {
                        transform(sub.begin(), sub.end(), sub.begin(), ::toupper);
                    }
                    else if (pattern == "lower")
                    {
                        transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
                    }
                    else if (pattern == "reverse")
                    {
                        reverse(sub.begin(), sub.end());
                    }
                    ans += sub;
                    i = j;
                }
            }
            else
            {
                ans += s[i];
            }
        }
        return ans;
    }
    /**
     * Note: 类名、方法名、参数名已经指定，请勿修改
     *
     *
     *
     * @param s string字符串
     * @return string字符串
     */
    string decodeStringWithPatterns(string s)
    {
        // write code here
        string ans = digui(s, 0, s.size() - 1);
        return ans;
    }
};

int main()
{
    Solution solution;
    string s = "aa1[#{upper}[a]2[bc]]ddd";
    string res = solution.decodeStringWithPatterns(s);
    cout << res << endl;
    return 0;
}
