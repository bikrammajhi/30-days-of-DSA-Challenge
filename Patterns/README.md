# Complete LeetCode Pattern Solutions in C++

## Quick Memory Guide 🧠

### Remember the patterns with this acronym: **"POTSFM OBTBD M"**
- **P**refix Sum, **O**verlapping Intervals, **T**wo Pointers, **S**liding Window
- **F**ast & Slow Pointers, **M**onotonic Stack  
- **O**verlapping Intervals, **B**inary Search, **T**ree Traversal, **B**FS, **D**FS
- **M**atrix, Backtracking, **D**ynamic Programming

### 2-Day Study Plan:
**Day 1**: Focus on patterns 1-8 (Arrays & Pointers)
**Day 2**: Focus on patterns 9-15 (Trees & Advanced)

---

## 1. Prefix Sum Pattern

### LeetCode #303: Range Sum Query - Immutable
```cpp
class NumArray {
private:
    vector<int> prefixSum;  // Store cumulative sums
    
public:
    NumArray(vector<int>& nums) {
        prefixSum.resize(nums.size() + 1, 0);  // Extra space for easier calculation
        
        // Build prefix sum array: prefixSum[i] = sum of nums[0] to nums[i-1]
        for(int i = 0; i < nums.size(); i++) {
            prefixSum[i + 1] = prefixSum[i] + nums[i];
        }
    }
    
    int sumRange(int left, int right) {
        // Sum from left to right = prefixSum[right+1] - prefixSum[left]
        return prefixSum[right + 1] - prefixSum[left];
    }
};
```

### LeetCode #560: Subarray Sum Equals K
```cpp
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefixSumCount;  // Store frequency of prefix sums
    prefixSumCount[0] = 1;  // Empty subarray has sum 0
    
    int count = 0, prefixSum = 0;
    
    for(int num : nums) {
        prefixSum += num;  // Running prefix sum
        
        // If (prefixSum - k) exists, we found subarrays with sum k
        if(prefixSumCount.find(prefixSum - k) != prefixSumCount.end()) {
            count += prefixSumCount[prefixSum - k];
        }
        
        prefixSumCount[prefixSum]++;  // Update frequency
    }
    
    return count;
}
```

---

## 2. Two Pointers Pattern

### LeetCode #167: Two Sum II - Input Array is Sorted
```cpp
vector<int> twoSum(vector<int>& numbers, int target) {
    int left = 0, right = numbers.size() - 1;
    
    while(left < right) {
        int sum = numbers[left] + numbers[right];
        
        if(sum == target) {
            return {left + 1, right + 1};  // 1-indexed answer
        }
        else if(sum < target) {
            left++;   // Need larger sum, move left pointer right
        }
        else {
            right--;  // Need smaller sum, move right pointer left
        }
    }
    
    return {};  // Should never reach here given problem constraints
}
```

### LeetCode #15: 3Sum
```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());  // Sort for two-pointer technique
    vector<vector<int>> result;
    
    for(int i = 0; i < nums.size() - 2; i++) {
        // Skip duplicates for first element
        if(i > 0 && nums[i] == nums[i-1]) continue;
        
        int left = i + 1, right = nums.size() - 1;
        int target = -nums[i];  // We want nums[left] + nums[right] = -nums[i]
        
        while(left < right) {
            int sum = nums[left] + nums[right];
            
            if(sum == target) {
                result.push_back({nums[i], nums[left], nums[right]});
                
                // Skip duplicates
                while(left < right && nums[left] == nums[left + 1]) left++;
                while(left < right && nums[right] == nums[right - 1]) right--;
                
                left++;
                right--;
            }
            else if(sum < target) {
                left++;
            }
            else {
                right--;
            }
        }
    }
    
    return result;
}
```

---

## 3. Sliding Window Pattern

### LeetCode #3: Longest Substring Without Repeating Characters
```cpp
int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;  // Track characters in current window
    int maxLength = 0;
    int left = 0;  // Left boundary of sliding window
    
    for(int right = 0; right < s.length(); right++) {
        // Shrink window from left until no duplicates
        while(window.count(s[right])) {
            window.erase(s[left]);
            left++;
        }
        
        window.insert(s[right]);  // Add current character
        maxLength = max(maxLength, right - left + 1);  // Update max length
    }
    
    return maxLength;
}
```

### LeetCode #76: Minimum Window Substring
```cpp
string minWindow(string s, string t) {
    unordered_map<char, int> need;  // Characters needed
    for(char c : t) need[c]++;
    
    int left = 0, right = 0;
    int valid = 0;  // Number of characters that satisfy the requirement
    int start = 0, minLen = INT_MAX;  // Result tracking
    
    while(right < s.length()) {
        char c = s[right];
        right++;
        
        // Update window data
        if(need.count(c)) {
            need[c]--;
            if(need[c] == 0) valid++;
        }
        
        // Try to shrink window from left
        while(valid == need.size()) {
            // Update result if current window is smaller
            if(right - left < minLen) {
                start = left;
                minLen = right - left;
            }
            
            char d = s[left];
            left++;
            
            if(need.count(d)) {
                if(need[d] == 0) valid--;
                need[d]++;
            }
        }
    }
    
    return minLen == INT_MAX ? "" : s.substr(start, minLen);
}
```

---

## 4. Fast & Slow Pointers Pattern

### LeetCode #141: Linked List Cycle
```cpp
bool hasCycle(ListNode *head) {
    if(!head || !head->next) return false;
    
    ListNode* slow = head;      // Tortoise moves 1 step
    ListNode* fast = head;      // Hare moves 2 steps
    
    while(fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        
        if(slow == fast) {      // They meet = cycle exists
            return true;
        }
    }
    
    return false;  // Fast reached end = no cycle
}
```

### LeetCode #287: Find the Duplicate Number
```cpp
int findDuplicate(vector<int>& nums) {
    // Phase 1: Find intersection point in the cycle
    int slow = nums[0];
    int fast = nums[0];
    
    do {
        slow = nums[slow];          // Move 1 step
        fast = nums[nums[fast]];    // Move 2 steps
    } while (slow != fast);
    
    // Phase 2: Find entrance to the cycle (duplicate number)
    slow = nums[0];
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    
    return slow;
}
```

---

## 5. LinkedList In-place Reversal Pattern

### LeetCode #206: Reverse Linked List
```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while(curr) {
        ListNode* next = curr->next;  // Store next node
        curr->next = prev;            // Reverse the link
        prev = curr;                  // Move prev forward
        curr = next;                  // Move curr forward
    }
    
    return prev;  // prev is the new head
}
```

### LeetCode #92: Reverse Linked List II
```cpp
ListNode* reverseBetween(ListNode* head, int left, int right) {
    if(!head || left == right) return head;
    
    ListNode dummy(0);
    dummy.next = head;
    ListNode* prev = &dummy;
    
    // Move to position before 'left'
    for(int i = 0; i < left - 1; i++) {
        prev = prev->next;
    }
    
    // Reverse the sublist
    ListNode* curr = prev->next;
    for(int i = 0; i < right - left; i++) {
        ListNode* next = curr->next;
        curr->next = next->next;
        next->next = prev->next;
        prev->next = next;
    }
    
    return dummy.next;
}
```

---

## 6. Monotonic Stack Pattern

### LeetCode #496: Next Greater Element I
```cpp
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
    unordered_map<int, int> nextGreater;  // num -> next greater element
    stack<int> st;  // Monotonic decreasing stack
    
    // Process nums2 to find next greater elements
    for(int num : nums2) {
        // Pop elements smaller than current num
        while(!st.empty() && st.top() < num) {
            nextGreater[st.top()] = num;
            st.pop();
        }
        st.push(num);
    }
    
    // Build result for nums1
    vector<int> result;
    for(int num : nums1) {
        result.push_back(nextGreater.count(num) ? nextGreater[num] : -1);
    }
    
    return result;
}
```

### LeetCode #739: Daily Temperatures
```cpp
vector<int> dailyTemperatures(vector<int>& temperatures) {
    vector<int> result(temperatures.size(), 0);
    stack<int> st;  // Store indices in decreasing order of temperatures
    
    for(int i = 0; i < temperatures.size(); i++) {
        // Pop indices with temperatures lower than current
        while(!st.empty() && temperatures[st.top()] < temperatures[i]) {
            int idx = st.top();
            st.pop();
            result[idx] = i - idx;  // Days to wait
        }
        st.push(i);
    }
    
    return result;
}
```

---

## 7. Top 'K' Elements Pattern

### LeetCode #215: Kth Largest Element in an Array
```cpp
int findKthLargest(vector<int>& nums, int k) {
    // Min heap to keep track of k largest elements
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for(int num : nums) {
        minHeap.push(num);
        
        // Keep only k largest elements
        if(minHeap.size() > k) {
            minHeap.pop();
        }
    }
    
    return minHeap.top();  // Kth largest element
}
```

### LeetCode #347: Top K Frequent Elements
```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    // Count frequencies
    unordered_map<int, int> freq;
    for(int num : nums) {
        freq[num]++;
    }
    
    // Min heap based on frequency
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;
    
    for(auto& p : freq) {
        minHeap.push({p.second, p.first});  // {frequency, number}
        
        if(minHeap.size() > k) {
            minHeap.pop();
        }
    }
    
    // Extract results
    vector<int> result;
    while(!minHeap.empty()) {
        result.push_back(minHeap.top().second);
        minHeap.pop();
    }
    
    return result;
}
```

---

## 8. Overlapping Intervals Pattern

### LeetCode #56: Merge Intervals
```cpp
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if(intervals.empty()) return {};
    
    // Sort by start time
    sort(intervals.begin(), intervals.end());
    
    vector<vector<int>> merged;
    merged.push_back(intervals[0]);
    
    for(int i = 1; i < intervals.size(); i++) {
        // If current interval overlaps with the last merged interval
        if(intervals[i][0] <= merged.back()[1]) {
            // Merge by extending the end time
            merged.back()[1] = max(merged.back()[1], intervals[i][1]);
        } else {
            // No overlap, add new interval
            merged.push_back(intervals[i]);
        }
    }
    
    return merged;
}
```

### LeetCode #57: Insert Interval
```cpp
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
    vector<vector<int>> result;
    int i = 0;
    
    // Add all intervals that end before newInterval starts
    while(i < intervals.size() && intervals[i][1] < newInterval[0]) {
        result.push_back(intervals[i]);
        i++;
    }
    
    // Merge overlapping intervals with newInterval
    while(i < intervals.size() && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = min(newInterval[0], intervals[i][0]);
        newInterval[1] = max(newInterval[1], intervals[i][1]);
        i++;
    }
    result.push_back(newInterval);
    
    // Add remaining intervals
    while(i < intervals.size()) {
        result.push_back(intervals[i]);
        i++;
    }
    
    return result;
}
```

---

## 9. Modified Binary Search Pattern

### LeetCode #33: Search in Rotated Sorted Array
```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while(left <= right) {
        int mid = left + (right - left) / 2;
        
        if(nums[mid] == target) return mid;
        
        // Check which half is sorted
        if(nums[left] <= nums[mid]) {  // Left half is sorted
            if(target >= nums[left] && target < nums[mid]) {
                right = mid - 1;  // Target in left half
            } else {
                left = mid + 1;   // Target in right half
            }
        } else {  // Right half is sorted
            if(target > nums[mid] && target <= nums[right]) {
                left = mid + 1;   // Target in right half
            } else {
                right = mid - 1;  // Target in left half
            }
        }
    }
    
    return -1;
}
```

### LeetCode #153: Find Minimum in Rotated Sorted Array
```cpp
int findMin(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    
    while(left < right) {
        int mid = left + (right - left) / 2;
        
        if(nums[mid] > nums[right]) {
            // Minimum is in right half
            left = mid + 1;
        } else {
            // Minimum is in left half (including mid)
            right = mid;
        }
    }
    
    return nums[left];
}
```

---

## 10. Binary Tree Traversal Pattern

### Tree Node Definition
```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

### LeetCode #94: Binary Tree Inorder Traversal
```cpp
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;
    
    while(curr || !st.empty()) {
        // Go to leftmost node
        while(curr) {
            st.push(curr);
            curr = curr->left;
        }
        
        // Process current node
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        
        // Move to right subtree
        curr = curr->right;
    }
    
    return result;
}
```

### LeetCode #230: Kth Smallest Element in a BST
```cpp
int kthSmallest(TreeNode* root, int k) {
    stack<TreeNode*> st;
    TreeNode* curr = root;
    
    while(curr || !st.empty()) {
        // Go to leftmost node
        while(curr) {
            st.push(curr);
            curr = curr->left;
        }
        
        // Process current node
        curr = st.top();
        st.pop();
        
        k--;  // Found one element in sorted order
        if(k == 0) return curr->val;
        
        curr = curr->right;
    }
    
    return -1;  // Should never reach here
}
```

---

## 11. Depth-First Search (DFS) Pattern

### LeetCode #133: Clone Graph
```cpp
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() { val = 0; neighbors = vector<Node*>(); }
    Node(int _val) { val = _val; neighbors = vector<Node*>(); }
};

Node* cloneGraph(Node* node) {
    if(!node) return nullptr;
    
    unordered_map<Node*, Node*> cloned;  // Original -> Clone mapping
    
    function<Node*(Node*)> dfs = [&](Node* original) -> Node* {
        if(cloned.count(original)) {
            return cloned[original];  // Already cloned
        }
        
        // Create clone
        Node* clone = new Node(original->val);
        cloned[original] = clone;
        
        // Clone all neighbors
        for(Node* neighbor : original->neighbors) {
            clone->neighbors.push_back(dfs(neighbor));
        }
        
        return clone;
    };
    
    return dfs(node);
}
```

### LeetCode #113: Path Sum II
```cpp
vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> result;
    vector<int> path;
    
    function<void(TreeNode*, int)> dfs = [&](TreeNode* node, int sum) {
        if(!node) return;
        
        path.push_back(node->val);
        sum -= node->val;
        
        // Check if it's a leaf and sum equals target
        if(!node->left && !node->right && sum == 0) {
            result.push_back(path);
        }
        
        // Continue DFS
        dfs(node->left, sum);
        dfs(node->right, sum);
        
        path.pop_back();  // Backtrack
    };
    
    dfs(root, targetSum);
    return result;
}
```

---

## 12. Breadth-First Search (BFS) Pattern

### LeetCode #102: Binary Tree Level Order Traversal
```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    if(!root) return {};
    
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while(!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        
        // Process all nodes at current level
        for(int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            
            currentLevel.push_back(node->val);
            
            // Add children for next level
            if(node->left) q.push(node->left);
            if(node->right) q.push(node->right);
        }
        
        result.push_back(currentLevel);
    }
    
    return result;
}
```

### LeetCode #994: Rotting Oranges
```cpp
int orangesRotting(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    queue<pair<int, int>> q;  // Store positions of rotten oranges
    int freshCount = 0;
    
    // Find all rotten oranges and count fresh ones
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(grid[i][j] == 2) {
                q.push({i, j});
            } else if(grid[i][j] == 1) {
                freshCount++;
            }
        }
    }
    
    if(freshCount == 0) return 0;  // No fresh oranges
    
    int minutes = 0;
    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    while(!q.empty()) {
        int size = q.size();
        bool hasNewRotten = false;
        
        // Process all currently rotten oranges
        for(int i = 0; i < size; i++) {
            auto [row, col] = q.front();
            q.pop();
            
            // Check all 4 directions
            for(auto [dr, dc] : directions) {
                int newRow = row + dr, newCol = col + dc;
                
                if(newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                   && grid[newRow][newCol] == 1) {
                    grid[newRow][newCol] = 2;  // Make it rotten
                    q.push({newRow, newCol});
                    freshCount--;
                    hasNewRotten = true;
                }
            }
        }
        
        if(hasNewRotten) minutes++;
    }
    
    return freshCount == 0 ? minutes : -1;
}
```

---

## 13. Matrix Traversal Pattern

### LeetCode #200: Number of Islands
```cpp
int numIslands(vector<vector<char>>& grid) {
    if(grid.empty()) return 0;
    
    int rows = grid.size(), cols = grid[0].size();
    int islands = 0;
    
    function<void(int, int)> dfs = [&](int row, int col) {
        // Boundary check and water check
        if(row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == '0') {
            return;
        }
        
        grid[row][col] = '0';  // Mark as visited
        
        // Visit all 4 directions
        dfs(row - 1, col);  // Up
        dfs(row + 1, col);  // Down
        dfs(row, col - 1);  // Left
        dfs(row, col + 1);  // Right
    };
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(grid[i][j] == '1') {
                islands++;
                dfs(i, j);  // Mark entire island as visited
            }
        }
    }
    
    return islands;
}
```

### LeetCode #733: Flood Fill
```cpp
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
    int originalColor = image[sr][sc];
    if(originalColor == color) return image;  // No change needed
    
    int rows = image.size(), cols = image[0].size();
    
    function<void(int, int)> dfs = [&](int row, int col) {
        // Boundary check and color check
        if(row < 0 || row >= rows || col < 0 || col >= cols 
           || image[row][col] != originalColor) {
            return;
        }
        
        image[row][col] = color;  // Fill with new color
        
        // Fill all 4 directions
        dfs(row - 1, col);
        dfs(row + 1, col);
        dfs(row, col - 1);
        dfs(row, col + 1);
    };
    
    dfs(sr, sc);
    return image;
}
```

---

## 14. Backtracking Pattern

### LeetCode #46: Permutations
```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;
    vector<bool> used(nums.size(), false);
    
    function<void()> backtrack = [&]() {
        // Base case: permutation is complete
        if(current.size() == nums.size()) {
            result.push_back(current);
            return;
        }
        
        // Try each unused number
        for(int i = 0; i < nums.size(); i++) {
            if(used[i]) continue;
            
            // Choose
            current.push_back(nums[i]);
            used[i] = true;
            
            // Explore
            backtrack();
            
            // Unchoose (backtrack)
            current.pop_back();
            used[i] = false;
        }
    };
    
    backtrack();
    return result;
}
```

### LeetCode #78: Subsets
```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;
    
    function<void(int)> backtrack = [&](int start) {
        // Add current subset to result
        result.push_back(current);
        
        // Try adding each remaining element
        for(int i = start; i < nums.size(); i++) {
            // Choose
            current.push_back(nums[i]);
            
            // Explore
            backtrack(i + 1);
            
            // Unchoose (backtrack)
            current.pop_back();
        }
    };
    
    backtrack(0);
    return result;
}
```

---

## 15. Dynamic Programming Patterns

### LeetCode #70: Climbing Stairs (Fibonacci Pattern)
```cpp
int climbStairs(int n) {
    if(n <= 2) return n;
    
    // dp[i] = number of ways to reach step i
    vector<int> dp(n + 1);
    dp[1] = 1;  // 1 way to reach step 1
    dp[2] = 2;  // 2 ways to reach step 2
    
    for(int i = 3; i <= n; i++) {
        // Can reach step i from step i-1 or step i-2
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}

// Space optimized version
int climbStairs(int n) {
    if(n <= 2) return n;
    
    int prev2 = 1, prev1 = 2;
    
    for(int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    
    return prev1;
}
```

### LeetCode #198: House Robber
```cpp
int rob(vector<int>& nums) {
    if(nums.empty()) return 0;
    if(nums.size() == 1) return nums[0];
    
    // dp[i] = maximum money that can be robbed up to house i
    vector<int> dp(nums.size());
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    
    for(int i = 2; i < nums.size(); i++) {
        // Either rob current house + money from i-2, or don't rob (take i-1)
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    
    return dp[nums.size() - 1];
}

// Space optimized
int rob(vector<int>& nums) {
    int prev2 = 0, prev1 = 0;
    
    for(int num : nums) {
        int curr = max(prev1, prev2 + num);
        prev2 = prev1;
        prev1 = curr;
    }
    
    return prev1;
}
```

### LeetCode #322: Coin Change
```cpp
int coinChange(vector<int>& coins, int amount) {
    // dp[i] = minimum coins needed to make amount i
    vector<int> dp(amount + 1, amount + 1);  // Initialize with impossible value
    dp[0] = 0;  // 0 coins needed for amount 0
    
    for(int i = 1; i <= amount; i++) {
        for(int coin : coins) {
            if(i >= coin) {
                // Use this coin if it gives better result
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    return dp[amount] > amount ? -1 : dp[amount];
}
```

### LeetCode #1143: Longest Common Subsequence
```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    
    // dp[i][j] = LCS length of text1[0..i-1] and text2[0..j-1]
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            if(text1[i - 1] == text2[j - 1]) {
                // Characters match, extend LCS
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                // Characters don't match, take maximum from either side
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
    return dp[m][n];
}
```

### LeetCode #416: Partition Equal Subset Sum (0/1 Knapsack Pattern)
```cpp
bool canPartition(vector<int>& nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    
    // If sum is odd, can't partition into equal subsets
    if(sum % 2 != 0) return false;
    
    int target = sum / 2;
    
    // dp[i] = true if sum i can be achieved with some subset
    vector<bool> dp(target + 1, false);
    dp[0] = true;  // Sum 0 is always achievable (empty subset)
    
    for(int num : nums) {
        // Traverse backwards to avoid using same element twice
        for(int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    
    return dp[target];
}
```

---

## 🎯 Interview Tips & Memory Techniques

### 1. Pattern Recognition Checklist
**Ask yourself these questions:**
- **Array/String + Multiple queries?** → Prefix Sum
- **Sorted array + Find pair/triplet?** → Two Pointers  
- **Substring/Subarray problems?** → Sliding Window
- **Linked List + Cycle?** → Fast & Slow Pointers
- **Next greater/smaller element?** → Monotonic Stack
- **Find top K elements?** → Heap/Priority Queue
- **Intervals overlap?** → Merge Intervals
- **Tree traversal?** → DFS/BFS
- **Find all solutions?** → Backtracking
- **Optimization problem?** → Dynamic Programming

### 2. Common Mistakes to Avoid
```cpp
// ❌ Wrong: Off-by-one errors
for(int i = 0; i < n; i++) {  // Should be i <= n for some problems
    
// ❌ Wrong: Not handling edge cases
if(nums.empty()) return 0;  // Always check empty inputs

// ❌ Wrong: Integer overflow
int sum = a + b;  // Use long long for large numbers

// ❌ Wrong: Not initializing variables
vector<int> dp(n);  // Should be dp(n, 0) or dp(n, -1)
```

### 3. Code Templates to Memorize

#### Binary Search Template
```cpp
int left = 0, right = n - 1;
while(left <= right) {
    int mid = left + (right - left) / 2;
    if(nums[mid] == target) return mid;
    else if(nums[mid] < target) left = mid + 1;
    else right = mid - 1;
}
return -1;
```

#### DFS Template
```cpp
void dfs(TreeNode* node) {
    if(!node) return;  // Base case
    
    // Process current node
    // ...
    
    dfs(node->left);   // Recurse left
    dfs(node->right);  // Recurse right
}
```

#### Backtracking Template
```cpp
void backtrack(/* parameters */) {
    if(/* base case */) {
        // Add to result
        return;
    }
    
    for(/* choices */) {
        // Make choice
        backtrack(/* updated parameters */);
        // Undo choice (if needed)
    }
}
```

### 4. Time & Space Complexity Quick Guide
- **Two Pointers**: O(n) time, O(1) space
- **Sliding Window**: O(n) time, O(1) space  
- **Binary Search**: O(log n) time, O(1) space
- **DFS/BFS**: O(V + E) time, O(V) space
- **Dynamic Programming**: Usually O(n²) time, O(n) space
- **Backtracking**: O(2^n) time in worst case

### 5. Last-Minute Review Strategy
**30 minutes before interview:**
1. Review the pattern recognition checklist
2. Practice drawing the key patterns on paper
3. Rehearse explaining your thought process out loud
4. Remember: It's okay to start with brute force and optimize!

### 6. During the Interview
1. **Clarify the problem** - Ask about edge cases, constraints
2. **Think out loud** - Explain your approach before coding
3. **Start simple** - Brute force first, then optimize
4. **Test your code** - Walk through with examples
5. **Analyze complexity** - Always mention time/space complexity
