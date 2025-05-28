# 30 Essential DSA Interview Patterns for GPU/CUDA Roles

## 1. Two Pointers

**Key Concept**: Use two pointers moving towards each other or in same direction to solve problems in O(n) time.

**Why Important for GPU Roles**: Demonstrates understanding of memory access patterns and optimization.

### Problems:
1. **Two Sum II - Input Array Is Sorted** (LeetCode 167)
2. **3Sum** (LeetCode 15)
3. **Container With Most Water** (LeetCode 11)
4. **Remove Duplicates from Sorted Array** (LeetCode 26)
5. **Trapping Rain Water** (LeetCode 42)

### Sample Solution - Implement Trie:
```cpp
class Trie {
private:
    struct TrieNode {
        unordered_map<char, TrieNode*> children;
        bool isEndOfWord;
        
        TrieNode() : isEndOfWord(false) {}
    };
    
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (curr->children.find(c) == curr->children.end()) {
                curr->children[c] = new TrieNode();
            }
            curr = curr->children[c];
        }
        curr->isEndOfWord = true;
    }
    
    bool search(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (curr->children.find(c) == curr->children.end()) {
                return false;
            }
            curr = curr->children[c];
        }
        return curr->isEndOfWord;
    }
    
    bool startsWith(string prefix) {
        TrieNode* curr = root;
        for (char c : prefix) {
            if (curr->children.find(c) == curr->children.end()) {
                return false;
            }
            curr = curr->children[c];
        }
        return true;
    }
};
```

## 2. Sliding Window

**Key Concept**: Maintain a window of elements and slide it to find optimal solutions.

**Why Important for GPU Roles**: Critical for understanding memory coalescing and data locality in GPU kernels.

### Problems:
1. **Longest Substring Without Repeating Characters** (LeetCode 3)
2. **Minimum Window Substring** (LeetCode 76)
3. **Sliding Window Maximum** (LeetCode 239)
4. **Longest Repeating Character Replacement** (LeetCode 424)
5. **Permutation in String** (LeetCode 567)

### Sample Solution - Longest Substring Without Repeating Characters:
```cpp
int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;
    int left = 0, maxLen = 0;
    
    for (int right = 0; right < s.length(); right++) {
        while (window.count(s[right])) {
            window.erase(s[left++]);
        }
        window.insert(s[right]);
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

## 3. Fast & Slow Pointers (Floyd's Cycle Detection)

**Key Concept**: Use pointers moving at different speeds to detect cycles or find middle elements.

**Why Important for GPU Roles**: Useful for memory management and detecting infinite loops in parallel algorithms.

### Problems:
1. **Linked List Cycle** (LeetCode 141)
2. **Find the Duplicate Number** (LeetCode 287)
3. **Happy Number** (LeetCode 202)
4. **Middle of the Linked List** (LeetCode 876)
5. **Linked List Cycle II** (LeetCode 142)

### Sample Solution - Linked List Cycle:
```cpp
bool hasCycle(ListNode *head) {
    if (!head || !head->next) return false;
    
    ListNode* slow = head;
    ListNode* fast = head->next;
    
    while (fast && fast->next) {
        if (slow == fast) return true;
        slow = slow->next;
        fast = fast->next->next;
    }
    return false;
}
```

## 4. Merge Intervals

**Key Concept**: Sort intervals and merge overlapping ones.

**Why Important for GPU Roles**: Essential for memory allocation and resource scheduling in GPU programming.

### Problems:
1. **Merge Intervals** (LeetCode 56)
2. **Insert Interval** (LeetCode 57)
3. **Non-overlapping Intervals** (LeetCode 435)
4. **Meeting Rooms II** (LeetCode 253)
5. **Interval List Intersections** (LeetCode 986)

### Sample Solution - Merge Intervals:
```cpp
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};
    
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> result = {intervals[0]};
    
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] <= result.back()[1]) {
            result.back()[1] = max(result.back()[1], intervals[i][1]);
        } else {
            result.push_back(intervals[i]);
        }
    }
    return result;
}
```

## 5. Cyclic Sort

**Key Concept**: Sort array by placing each element at its correct index.

**Why Important for GPU Roles**: Demonstrates understanding of data placement and indexing strategies.

### Problems:
1. **Missing Number** (LeetCode 268)
2. **Find All Numbers Disappeared in an Array** (LeetCode 448)
3. **Find the Duplicate Number** (LeetCode 287)
4. **First Missing Positive** (LeetCode 41)
5. **Find All Duplicates in an Array** (LeetCode 442)

### Sample Solution - Missing Number:
```cpp
int missingNumber(vector<int>& nums) {
    int n = nums.size();
    int expectedSum = n * (n + 1) / 2;
    int actualSum = 0;
    
    for (int num : nums) {
        actualSum += num;
    }
    
    return expectedSum - actualSum;
}
```

## 6. In-place Reversal of LinkedList

**Key Concept**: Reverse parts of linked list without extra space.

**Why Important for GPU Roles**: Shows pointer manipulation skills crucial for kernel memory management.

### Problems:
1. **Reverse Linked List** (LeetCode 206)
2. **Reverse Linked List II** (LeetCode 92)
3. **Reverse Nodes in k-Group** (LeetCode 25)
4. **Swap Nodes in Pairs** (LeetCode 24)
5. **Palindrome Linked List** (LeetCode 234)

### Sample Solution - Reverse Linked List:
```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* current = head;
    
    while (current) {
        ListNode* next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }
    
    return prev;
}
```

## 7. Tree Breadth First Search (BFS)

**Key Concept**: Level-order traversal using queue.

**Why Important for GPU Roles**: Understanding of parallel traversal patterns and work distribution.

### Problems:
1. **Binary Tree Level Order Traversal** (LeetCode 102)
2. **Binary Tree Zigzag Level Order Traversal** (LeetCode 103)
3. **Minimum Depth of Binary Tree** (LeetCode 111)
4. **Binary Tree Right Side View** (LeetCode 199)
5. **Average of Levels in Binary Tree** (LeetCode 637)

### Sample Solution - Level Order Traversal:
```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

## 8. Tree Depth First Search (DFS)

**Key Concept**: Recursive or stack-based tree traversal.

**Why Important for GPU Roles**: Fundamental for understanding recursion limits and stack management in GPU kernels.

### Problems:
1. **Path Sum** (LeetCode 112)
2. **Path Sum II** (LeetCode 113)
3. **Sum Root to Leaf Numbers** (LeetCode 129)
4. **Binary Tree Maximum Path Sum** (LeetCode 124)
5. **Diameter of Binary Tree** (LeetCode 543)

### Sample Solution - Path Sum:
```cpp
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;
    
    if (!root->left && !root->right) {
        return root->val == targetSum;
    }
    
    return hasPathSum(root->left, targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
}
```

## 9. Two Heaps

**Key Concept**: Use min and max heaps to solve problems efficiently.

**Why Important for GPU Roles**: Understanding of priority-based scheduling and resource allocation.

### Problems:
1. **Find Median from Data Stream** (LeetCode 295)
2. **Sliding Window Median** (LeetCode 480)
3. **IPO** (LeetCode 502)
4. **Find Right Interval** (LeetCode 436)
5. **Maximum Performance of a Team** (LeetCode 1383)

### Sample Solution - Find Median from Data Stream:
```cpp
class MedianFinder {
private:
    priority_queue<int> maxHeap; // for smaller half
    priority_queue<int, vector<int>, greater<int>> minHeap; // for larger half
    
public:
    void addNum(int num) {
        if (maxHeap.empty() || num <= maxHeap.top()) {
            maxHeap.push(num);
        } else {
            minHeap.push(num);
        }
        
        // Balance heaps
        if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        } else if (minHeap.size() > maxHeap.size() + 1) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            return (maxHeap.top() + minHeap.top()) / 2.0;
        }
        return maxHeap.size() > minHeap.size() ? maxHeap.top() : minHeap.top();
    }
};
```

## 10. Subsets

**Key Concept**: Generate all possible subsets using backtracking.

**Why Important for GPU Roles**: Demonstrates combinatorial optimization skills useful for parallel algorithm design.

### Problems:
1. **Subsets** (LeetCode 78)
2. **Subsets II** (LeetCode 90)
3. **Permutations** (LeetCode 46)
4. **Permutations II** (LeetCode 47)
5. **Combination Sum** (LeetCode 39)

### Sample Solution - Subsets:
```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;
    
    function<void(int)> backtrack = [&](int start) {
        result.push_back(current);
        
        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(i + 1);
            current.pop_back();
        }
    };
    
    backtrack(0);
    return result;
}
```

## 11. Modified Binary Search

**Key Concept**: Adapt binary search for rotated arrays, peak finding, etc.

**Why Important for GPU Roles**: Critical for efficient data access patterns and optimization in parallel algorithms.

### Problems:
1. **Search in Rotated Sorted Array** (LeetCode 33)
2. **Find Minimum in Rotated Sorted Array** (LeetCode 153)
3. **Search a 2D Matrix** (LeetCode 74)
4. **Find Peak Element** (LeetCode 162)
5. **Search in Rotated Sorted Array II** (LeetCode 81)

### Sample Solution - Search in Rotated Sorted Array:
```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) return mid;
        
        if (nums[left] <= nums[mid]) { // Left side is sorted
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else { // Right side is sorted
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}
```

## 12. Bitwise XOR

**Key Concept**: Use XOR properties to solve problems efficiently.

**Why Important for GPU Roles**: Bit manipulation is crucial for GPU optimization and parallel algorithms.

### Problems:
1. **Single Number** (LeetCode 136)
2. **Single Number II** (LeetCode 137)
3. **Single Number III** (LeetCode 260)
4. **Missing Number** (LeetCode 268)
5. **Find the Difference** (LeetCode 389)

### Sample Solution - Single Number:
```cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

## 13. Top K Elements

**Key Concept**: Find top K elements using heaps or quickselect.

**Why Important for GPU Roles**: Essential for parallel reduction operations and priority-based processing.

### Problems:
1. **Kth Largest Element in an Array** (LeetCode 215)
2. **Top K Frequent Elements** (LeetCode 347)
3. **K Closest Points to Origin** (LeetCode 973)
4. **Kth Smallest Element in a Sorted Matrix** (LeetCode 378)
5. **Find K Pairs with Smallest Sums** (LeetCode 373)

### Sample Solution - Kth Largest Element:
```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }
    
    return minHeap.top();
}
```

## 14. K-way Merge

**Key Concept**: Merge K sorted arrays/lists efficiently.

**Why Important for GPU Roles**: Fundamental for understanding merge operations in parallel sorting algorithms.

### Problems:
1. **Merge k Sorted Lists** (LeetCode 23)
2. **Kth Smallest Element in a Sorted Matrix** (LeetCode 378)
3. **Smallest Range Covering Elements from K Lists** (LeetCode 632)
4. **Find K Pairs with Smallest Sums** (LeetCode 373)
5. **Merge Sorted Array** (LeetCode 88)

### Sample Solution - Merge k Sorted Lists:
```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    if (lists.empty()) return nullptr;
    
    auto cmp = [](ListNode* a, ListNode* b) {
        return a->val > b->val;
    };
    
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
    
    for (ListNode* list : lists) {
        if (list) pq.push(list);
    }
    
    ListNode dummy(0);
    ListNode* tail = &dummy;
    
    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();
        
        tail->next = node;
        tail = tail->next;
        
        if (node->next) {
            pq.push(node->next);
        }
    }
    
    return dummy.next;
}
```

## 15. 0/1 Knapsack (Dynamic Programming)

**Key Concept**: Optimize selection with weight/capacity constraints.

**Why Important for GPU Roles**: Models resource allocation problems common in GPU memory management.

### Problems:
1. **Partition Equal Subset Sum** (LeetCode 416)
2. **Target Sum** (LeetCode 494)
3. **Last Stone Weight II** (LeetCode 1049)
4. **Ones and Zeroes** (LeetCode 474)
5. **Coin Change** (LeetCode 322)

### Sample Solution - Partition Equal Subset Sum:
```cpp
bool canPartition(vector<int>& nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2) return false;
    
    int target = sum / 2;
    vector<bool> dp(target + 1, false);
    dp[0] = true;
    
    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    
    return dp[target];
}
```

## 16. Unbounded Knapsack

**Key Concept**: Unlimited use of items in knapsack problems.

**Why Important for GPU Roles**: Models scenarios with unlimited resources in parallel processing.

### Problems:
1. **Coin Change** (LeetCode 322)
2. **Coin Change 2** (LeetCode 518)
3. **Perfect Squares** (LeetCode 279)
4. **Minimum Cost For Tickets** (LeetCode 983)
5. **Combination Sum IV** (LeetCode 377)

### Sample Solution - Coin Change:
```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (i >= coin) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    return dp[amount] > amount ? -1 : dp[amount];
}
```

## 17. Fibonacci Numbers

**Key Concept**: Problems following Fibonacci-like recurrence relations.

**Why Important for GPU Roles**: Base case for understanding dynamic programming optimization.

### Problems:
1. **Climbing Stairs** (LeetCode 70)
2. **House Robber** (LeetCode 198)
3. **House Robber II** (LeetCode 213)
4. **Fibonacci Number** (LeetCode 509)
5. **Min Cost Climbing Stairs** (LeetCode 746)

### Sample Solution - Climbing Stairs:
```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    
    int prev2 = 1, prev1 = 2;
    
    for (int i = 3; i <= n; i++) {
        int current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}
```

## 18. Palindromic Subsequence

**Key Concept**: Find palindromes in strings or subsequences.

**Why Important for GPU Roles**: Demonstrates string processing skills important for text processing kernels.

### Problems:
1. **Longest Palindromic Substring** (LeetCode 5)
2. **Palindromic Substrings** (LeetCode 647)
3. **Longest Palindromic Subsequence** (LeetCode 516)
4. **Valid Palindrome** (LeetCode 125)
5. **Palindrome Partitioning** (LeetCode 131)

### Sample Solution - Longest Palindromic Substring:
```cpp
string longestPalindrome(string s) {
    if (s.empty()) return "";
    
    int start = 0, maxLen = 1;
    
    auto expandAroundCenter = [&](int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            int len = right - left + 1;
            if (len > maxLen) {
                start = left;
                maxLen = len;
            }
            left--;
            right++;
        }
    };
    
    for (int i = 0; i < s.length(); i++) {
        expandAroundCenter(i, i);       // odd length
        expandAroundCenter(i, i + 1);   // even length
    }
    
    return s.substr(start, maxLen);
}
```

## 19. Longest Common Subsequence

**Key Concept**: Find common subsequences between strings.

**Why Important for GPU Roles**: Important for sequence alignment algorithms used in bioinformatics and ML.

### Problems:
1. **Longest Common Subsequence** (LeetCode 1143)
2. **Edit Distance** (LeetCode 72)
3. **Distinct Subsequences** (LeetCode 115)
4. **Shortest Common Supersequence** (LeetCode 1092)
5. **Delete Operation for Two Strings** (LeetCode 583)

### Sample Solution - Longest Common Subsequence:
```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    return dp[m][n];
}
```

## 20. Topological Sort (Graph)

**Key Concept**: Linear ordering of vertices in a directed acyclic graph.

**Why Important for GPU Roles**: Essential for dependency resolution and task scheduling in parallel systems.

### Problems:
1. **Course Schedule** (LeetCode 207)
2. **Course Schedule II** (LeetCode 210)
3. **Alien Dictionary** (LeetCode 269)
4. **Minimum Height Trees** (LeetCode 310)
5. **Parallel Courses** (LeetCode 1136)

### Sample Solution - Course Schedule:
```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> indegree(numCourses, 0);
    
    for (auto& prereq : prerequisites) {
        graph[prereq[1]].push_back(prereq[0]);
        indegree[prereq[0]]++;
    }
    
    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    
    int completed = 0;
    while (!q.empty()) {
        int course = q.front();
        q.pop();
        completed++;
        
        for (int next : graph[course]) {
            indegree[next]--;
            if (indegree[next] == 0) {
                q.push(next);
            }
        }
    }
    
    return completed == numCourses;
}
```

## 21. Union Find

**Key Concept**: Efficiently manage disjoint sets with union and find operations.

**Why Important for GPU Roles**: Crucial for parallel connectivity and clustering algorithms.

### Problems:
1. **Number of Islands** (LeetCode 200)
2. **Accounts Merge** (LeetCode 721)
3. **Redundant Connection** (LeetCode 684)
4. **Most Stones Removed with Same Row or Column** (LeetCode 947)
5. **Number of Provinces** (LeetCode 547)

### Sample Solution - Number of Islands:
```cpp
class UnionFind {
public:
    vector<int> parent, rank;
    int count;
    
    UnionFind(int n) : parent(n), rank(n, 0), count(0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        count--;
    }
    
    void setCount(int n) { count = n; }
};

int numIslands(vector<vector<char>>& grid) {
    if (grid.empty()) return 0;
    
    int m = grid.size(), n = grid[0].size();
    UnionFind uf(m * n);
    
    int islands = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                islands++;
            }
        }
    }
    uf.setCount(islands);
    
    vector<pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                for (auto& dir : directions) {
                    int ni = i + dir.first, nj = j + dir.second;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == '1') {
                        uf.unite(i * n + j, ni * n + nj);
                    }
                }
            }
        }
    }
    
    return uf.count;
}
```

## 22. Ordered Set

**Key Concept**: Maintain sorted order while supporting insertions and deletions.

**Why Important for GPU Roles**: Important for maintaining sorted data structures in parallel algorithms.

### Problems:
1. **Contains Duplicate III** (LeetCode 220)
2. **The Skyline Problem** (LeetCode 218)
3. **Falling Squares** (LeetCode 699)
4. **My Calendar I** (LeetCode 729)
5. **My Calendar III** (LeetCode 732)

### Sample Solution - Contains Duplicate III:
```cpp
bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
    if (t < 0) return false;
    
    set<long> window;
    
    for (int i = 0; i < nums.size(); i++) {
        if (i > k) {
            window.erase(nums[i - k - 1]);
        }
        
        auto it = window.lower_bound((long)nums[i] - t);
        if (it != window.end() && *it <= (long)nums[i] + t) {
            return true;
        }
        
        window.insert(nums[i]);
    }
    
    return false;
}
```

## 23. Multi-threaded

**Key Concept**: Coordinate multiple threads safely and efficiently.

**Why Important for GPU Roles**: Directly relevant for understanding parallel execution and synchronization.

### Problems:
1. **Print in Order** (LeetCode 1114)
2. **Print FooBar Alternately** (LeetCode 1115)
3. **Print Zero Even Odd** (LeetCode 1116)
4. **Building H2O** (LeetCode 1117)
5. **The Dining Philosophers** (LeetCode 1226)

### Sample Solution - Print in Order:
```cpp
class Foo {
private:
    mutex m1, m2;
    
public:
    Foo() {
        m1.lock();
        m2.lock();
    }

    void first(function<void()> printFirst) {
        printFirst();
        m1.unlock();
    }

    void second(function<void()> printSecond) {
        m1.lock();
        printSecond();
        m1.unlock();
        m2.unlock();
    }

    void third(function<void()> printThird) {
        m2.lock();
        printThird();
        m2.unlock();
    }
};
```

## 24. Monotonic Stack

**Key Concept**: Maintain stack in monotonic order to solve range queries efficiently.

**Why Important for GPU Roles**: Useful for optimization problems and maintaining sorted data in parallel processing.

### Problems:
1. **Next Greater Element I** (LeetCode 496)
2. **Daily Temperatures** (LeetCode 739)
3. **Largest Rectangle in Histogram** (LeetCode 84)
4. **Trapping Rain Water** (LeetCode 42)
5. **Remove K Digits** (LeetCode 402)

### Sample Solution - Daily Temperatures:
```cpp
vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st; // stores indices
    
    for (int i = 0; i < n; i++) {
        while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
            int idx = st.top();
            st.pop();
            result[idx] = i - idx;
        }
        st.push(i);
    }
    
    return result;
}
```

## 25. Trie (Prefix Tree)

**Key Concept**: Tree data structure for efficient string operations.

**Why Important for GPU Roles**: Useful for string matching algorithms and autocomplete features in GPU applications.

### Problems:
1. **Implement Trie (Prefix Tree)** (LeetCode 208)
2. **Word Search II** (LeetCode 212)
3. **Design Add and Search Words Data Structure** (LeetCode 211)
4. **Replace Words** (LeetCode 648)
5. **Word Break** (LeetCode 139)

### Sample

```cpp
class TrieNode {
public:
    TrieNode* children[26];
    bool isEndOfWord;

    TrieNode() {
        isEndOfWord = false;
        for(int i = 0; i < 26; ++i)
            children[i] = nullptr;
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    // Insert a word into the trie.
    void insert(string word) {
        TrieNode* node = root;
        for(char c : word) {
            int index = c - 'a';
            if(node->children[index] == nullptr)
                node->children[index] = new TrieNode();
            node = node->children[index];
        }
        node->isEndOfWord = true;
    }

    // Returns true if the word is in the trie.
    bool search(string word) {
        TrieNode* node = root;
        for(char c : word) {
            int index = c - 'a';
            if(node->children[index] == nullptr)
                return false;
            node = node->children[index];
        }
        return node->isEndOfWord;
    }

    // Returns true if there is any word in the trie that starts with the given prefix.
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for(char c : prefix) {
            int index = c - 'a';
            if(node->children[index] == nullptr)
                return false;
            node = node->children[index];
        }
        return true;
    }
};


```


## 26. Hash Maps

**Key Concept**: Use hash maps for O(1) lookups and frequency counting.

**Why Important for GPU Roles**: Understanding hash-based data structures is crucial for efficient parallel algorithms.

### Problems:
1. **Two Sum** (LeetCode 1)
2. **Group Anagrams** (LeetCode 49)
3. **Valid Anagram** (LeetCode 242)
4. **Longest Substring Without Repeating Characters** (LeetCode 3)
5. **First Unique Character in a String** (LeetCode 387)

### Sample Solution - Group Anagrams:
```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;
    
    for (string& str : strs) {
        string key = str;
        sort(key.begin(), key.end());
        groups[key].push_back(str);
    }
    
    vector<vector<string>> result;
    for (auto& group : groups) {
        result.push_back(move(group.second));
    }
    
    return result;
}
```

## 27. Intervals

**Key Concept**: Work with time intervals, scheduling, and overlapping ranges.

**Why Important for GPU Roles**: Critical for resource scheduling and memory management in GPU kernels.

### Problems:
1. **Meeting Rooms** (LeetCode 252)
2. **Meeting Rooms II** (LeetCode 253)
3. **Minimum Number of Arrows to Burst Balloons** (LeetCode 452)
4. **Non-overlapping Intervals** (LeetCode 435)
5. **Car Pooling** (LeetCode 1094)

### Sample Solution - Meeting Rooms II:
```cpp
int minMeetingRooms(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;
    
    vector<int> starts, ends;
    for (auto& interval : intervals) {
        starts.push_back(interval[0]);
        ends.push_back(interval[1]);
    }
    
    sort(starts.begin(), starts.end());
    sort(ends.begin(), ends.end());
    
    int rooms = 0, maxRooms = 0;
    int i = 0, j = 0;
    
    while (i < starts.size()) {
        if (starts[i] < ends[j]) {
            rooms++;
            i++;
        } else {
            rooms--;
            j++;
        }
        maxRooms = max(maxRooms, rooms);
    }
    
    return maxRooms;
}
```

## 28. Greedy

**Key Concept**: Make locally optimal choices to find global optimal solution.

**Why Important for GPU Roles**: Essential for optimization problems and resource allocation in parallel systems.

### Problems:
1. **Jump Game** (LeetCode 55)
2. **Jump Game II** (LeetCode 45)
3. **Gas Station** (LeetCode 134)
4. **Task Scheduler** (LeetCode 621)
5. **Minimum Number of Arrows to Burst Balloons** (LeetCode 452)

### Sample Solution - Jump Game:
```cpp
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    
    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
        if (maxReach >= nums.size() - 1) return true;
    }
    
    return true;
}
```

## 29. Backtracking

**Key Concept**: Explore all possible solutions by trying partial solutions and abandoning them if they can't lead to a complete solution.

**Why Important for GPU Roles**: Understanding of exploration algorithms and pruning strategies for parallel search.

### Problems:
1. **N-Queens** (LeetCode 51)
2. **Sudoku Solver** (LeetCode 37)
3. **Word Search** (LeetCode 79)
4. **Palindrome Partitioning** (LeetCode 131)
5. **Generate Parentheses** (LeetCode 22)

### Sample Solution - Generate Parentheses:
```cpp
vector<string> generateParenthesis(int n) {
    vector<string> result;
    string current;
    
    function<void(int, int)> backtrack = [&](int open, int close) {
        if (current.length() == 2 * n) {
            result.push_back(current);
            return;
        }
        
        if (open < n) {
            current.push_back('(');
            backtrack(open + 1, close);
            current.pop_back();
        }
        
        if (close < open) {
            current.push_back(')');
            backtrack(open, close + 1);
            current.pop_back();
        }
    };
    
    backtrack(0, 0);
    return result;
}
```

## 30. Dynamic Programming on Trees

**Key Concept**: Apply DP concepts to tree structures for optimization problems.

**Why Important for GPU Roles**: Advanced pattern showing complex recursive optimization, relevant for hierarchical algorithms in GPU computing.

### Problems:
1. **House Robber III** (LeetCode 337)
2. **Binary Tree Maximum Path Sum** (LeetCode 124)
3. **Diameter of Binary Tree** (LeetCode 543)
4. **Longest Univalue Path** (LeetCode 687)
5. **Binary Tree Cameras** (LeetCode 968)

### Sample Solution - House Robber III:
```cpp
int rob(TreeNode* root) {
    function<pair<int, int>(TreeNode*)> dfs = [&](TreeNode* node) -> pair<int, int> {
        if (!node) return {0, 0};
        
        auto left = dfs(node->left);
        auto right = dfs(node->right);
        
        // {rob current node, don't rob current node}
        int robCurrent = node->val + left.second + right.second;
        int skipCurrent = max(left.first, left.second) + max(right.first, right.second);
        
        return {robCurrent, skipCurrent};
    };
    
    auto result = dfs(root);
    return max(result.first, result.second);
}
```

---

## Study Strategy for GPU/CUDA Interviews

### Phase 1: Foundation (Weeks 1-2)
- Master Two Pointers, Sliding Window, and Fast/Slow Pointers
- Focus on memory access patterns and optimization thinking
- Practice explaining time/space complexity trade-offs

### Phase 2: Core Algorithms (Weeks 3-4)
- Tree traversals (BFS/DFS) - crucial for parallel algorithms
- Binary Search variations - essential for efficient data access
- Hash Maps and basic DP patterns

### Phase 3: Advanced Patterns (Weeks 5-6)
- Union Find and Topological Sort - important for parallel connectivity
- Heaps and Priority Queues - relevant for GPU scheduling
- Bit manipulation and XOR patterns

### Phase 4: Specialization (Weeks 7-8)
- Multi-threading problems - directly applicable to GPU programming
- Greedy algorithms - optimization mindset
- Advanced DP and Backtracking

### Interview Tips for GPU/CUDA Roles:

1. **Always discuss optimization**: Mention time/space complexity and potential optimizations
2. **Think in parallel**: When possible, discuss how algorithms could be parallelized
3. **Memory access patterns**: Show awareness of cache efficiency and memory coalescing
4. **Scalability**: Discuss how solutions scale with input size
5. **Trade-offs**: Always mention trade-offs between different approaches

### Company-Specific Focus:

**NVIDIA/AMD**: Heavy emphasis on parallel algorithms, memory optimization, and mathematical computations

**OpenAI/DeepMind**: Focus on ML-related algorithms, matrix operations, and optimization problems

**Meta/Google**: Balanced approach with system design thinking and scalability discussions

**Apple**: Emphasis on efficiency, battery optimization, and real-time processing

Remember: The key is not just solving problems correctly, but demonstrating the optimization mindset and parallel thinking that's crucial for GPU kernel development roles. Solution - Two Sum II:
```cpp
vector<int> twoSum(vector<int>& numbers, int target) {
    int left = 0, right = numbers.size() - 1;
    while (left < right) {
        int sum = numbers[left] + numbers[right];
        if (sum == target) return {left + 1, right + 1};
        else if (sum < target) left++;
        else right--;
    }
    return {};
}
```
