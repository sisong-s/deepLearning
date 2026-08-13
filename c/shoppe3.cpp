#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    /**
     * Note: 类名、方法名、参数名已经指定，请勿修改
     *
     *
     *
     * @param input_info string字符串
     * @return string字符串vector
     */
    vector<string> solution(string input_info)
    {
        // write code here
        vector<string> res;

        return res;
    }
};

int main()
{
    string s = "10;5,A,B,C,AA,F,E,DD;8,A,CC,EE;3,C,D,E,AA,BB,CC,EE,F";
    Solution solution;
    vector<string> res = solution.solution(s);
    for (const auto &item : res)
    {
        cout << item << endl;
    }
    return 0;
}