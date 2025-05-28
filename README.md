# 30-days-of-DSA-Challenge
# 🧠 DSA Patterns Memory Tricks & Shortcuts

A comprehensive guide with visual mnemonics, patterns, and shortcuts to master 30 essential DSA interview patterns.

---

## 📝 Quick Pattern Recognition Map

| **Problem Type** | **Keywords** | **Pattern** | **Visual Cue** |
|------------------|--------------|-------------|-----------------|
| Array/String with window | "substring", "subarray", "consecutive" | Sliding Window | 🪟 Window sliding |
| Two elements sum/diff | "two sum", "pair", "complement" | Two Pointers | ⬅️➡️ Arrows meeting |
| Cycle detection | "cycle", "duplicate", "loop" | Fast/Slow Pointers | 🐢🐰 Tortoise & Hare |
| Overlapping ranges | "intervals", "meetings", "schedule" | Merge Intervals | 📊 Timeline merge |
| Find missing/duplicate | "missing", "duplicate", range [1,n] | Cyclic Sort | 🔄 Circle arrangement |
| Reverse in groups | "reverse", "k groups", "pairs" | In-place Reversal | ↩️ Arrow flip |
| Level by level | "level order", "by levels", "width" | Tree BFS | 🌊 Wave pattern |
| Root to leaf | "path sum", "root to leaf" | Tree DFS | 🌳 Branch diving |
| Find median/extremes | "median", "kth largest/smallest" | Two Heaps | ⚖️ Balance scale |
| All combinations | "subsets", "permutations", "combinations" | Subsets/Backtrack | 🌟 Star explosion |

---

## 🎯 Pattern-Specific Tricks

### 1. Two Pointers 👥
**Memory Trick**: "Meet in the Middle" or "Chase Each Other"

**Visual**: `←---🎯---→` (arrows moving towards target)

**When to Use**:
- Sorted array + find pair/triplet
- Remove duplicates
- Palindrome check

**Code Pattern**:
```cpp
int left = 0, right = n - 1;
while (left < right) {
    // Process condition
    if (condition) return result;
    else if (sum < target) left++;
    else right--;
}
```

**Memory Hook**: "LEFT goes RIGHT, RIGHT goes LEFT until they MEET"

---

### 2. Sliding Window 🪟
**Memory Trick**: "Window Shopping" - expand to find, shrink to optimize

**Visual**: `[====🪟====]` (window sliding over array)

**Templates**:

**Fixed Size Window**:
```cpp
for (int i = 0; i < k; i++) // Build initial window
for (int i = k; i < n; i++) // Slide window
```

**Variable Size Window**:
```cpp
int left = 0;
for (int right = 0; right < n; right++) {
    // Expand window
    while (invalid_condition) {
        // Shrink window
        left++;
    }
    // Update answer
}
```

**Memory Hook**: "EXPAND when happy, SHRINK when sad"

---

### 3. Fast & Slow Pointers 🐢🐰
**Memory Trick**: "Tortoise and Hare Race"

**Visual**: 
```
🐢 → → → 🐰
slow    fast (2x speed)
```

**Pattern**:
```cpp
ListNode* slow = head;
ListNode* fast = head;
while (fast && fast->next) {
    slow = slow->next;      // 1 step
    fast = fast->next->next; // 2 steps
    if (slow == fast) return true; // Cycle found
}
```

**Memory Hook**: "When they MEET, there's a CYCLE to beat!"

---

### 4. Merge Intervals 📊
**Memory Trick**: "Timeline Merger"

**Visual**: 
```
Before: |-----|   |-----|     |--|
After:  |-----------|           |--|
```

**Steps** (Remember: **S.M.C**):
1. **S**ort intervals by start time
2. **M**erge overlapping intervals  
3. **C**heck: if `current.start <= previous.end` → merge

**Pattern**:
```cpp
sort(intervals.begin(), intervals.end());
vector<vector<int>> result = {intervals[0]};
for (int i = 1; i < intervals.size(); i++) {
    if (intervals[i][0] <= result.back()[1]) {
        result.back()[1] = max(result.back()[1], intervals[i][1]);
    } else {
        result.push_back(intervals[i]);
    }
}
```

---

### 5. Cyclic Sort 🔄
**Memory Trick**: "Everyone Goes Home" - each number goes to its rightful place

**Visual**: 
```
Index: 0  1  2  3  4
Value: 3  1  4  2  0
       ↓  ↓  ↓  ↓  ↓
Home:  0  1  2  3  4
```

**Key Insight**: If array has numbers 1 to n, number `x` belongs at index `x-1`

**Pattern**:
```cpp
for (int i = 0; i < n; i++) {
    while (nums[i] != i + 1 && nums[i] <= n) {
        swap(nums[i], nums[nums[i] - 1]);
    }
}
```

**Memory Hook**: "SWAP until everyone is HOME!"

---

### 6. In-place LinkedList Reversal ↩️
**Memory Trick**: "Three Dancers" - prev, curr, next

**Visual**:
```
prev  curr  next
 ↑     ↑     ↑
null   1  →  2  →  3  →  null
      ↗
```

**Pattern**:
```cpp
ListNode* prev = nullptr;
ListNode* curr = head;
while (curr) {
    ListNode* next = curr->next; // Save next
    curr->next = prev;           // Reverse link
    prev = curr;                 // Move prev
    curr = next;                 // Move curr
}
return prev; // New head
```

**Memory Hook**: "Save, Reverse, Move, Move"

---

### 7. Tree BFS 🌊
**Memory Trick**: "Water Wave" - level spreads like water

**Visual**:
```
      1      ← Level 0
    /   \
   2     3    ← Level 1  
  / \   / \
 4   5 6   7  ← Level 2
```

**Pattern**:
```cpp
queue<TreeNode*> q;
q.push(root);
while (!q.empty()) {
    int size = q.size(); // Current level size
    for (int i = 0; i < size; i++) {
        TreeNode* node = q.front(); q.pop();
        // Process node
        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
    }
}
```

**Memory Hook**: "Queue the LEVEL, process the WAVE"

---

### 8. Tree DFS 🌳
**Memory Trick**: "Root to Leaf Adventure"

**Visual**:
```
    🌲 Root
   /  \
  🌿   🌿 (Go deep first)
 /    /  \
🍃   🍃   🍃 Leaves
```

**Three Types**:
- **Preorder**: Root → Left → Right (📖 "Read book order")
- **Inorder**: Left → Root → Right (🔢 "Sorted for BST")  
- **Postorder**: Left → Right → Root (🧹 "Clean up after kids")

**Recursive Pattern**:
```cpp
void dfs(TreeNode* node, /* parameters */) {
    if (!node) return;
    
    // Preorder: process here
    dfs(node->left, /* params */);
    // Inorder: process here
    dfs(node->right, /* params */);
    // Postorder: process here
}
```

---

### 9. Two Heaps ⚖️
**Memory Trick**: "Balance Scale" - MaxHeap (left) vs MinHeap (right)

**Visual**:
```
MaxHeap     MinHeap
[4,3,2] ⚖️ [5,6,7]
 (left)     (right)
```

**Use Case**: Find median from data stream

**Pattern**:
```cpp
priority_queue<int> maxHeap; // Left half (smaller elements)
priority_queue<int, vector<int>, greater<int>> minHeap; // Right half

// Balance rule: |maxHeap.size() - minHeap.size()| ≤ 1
```

**Memory Hook**: "MAX on left, MIN on right, keep BALANCED"

---

### 10. Subsets/Backtracking 🌟
**Memory Trick**: "Decision Tree" - include or exclude each element

**Visual**:
```
        []
       /  \
    [1]    []
   /  \   /  \
[1,2]  [1] [2] []
```

**Pattern**:
```cpp
void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) {
    result.push_back(current); // Add current subset
    
    for (int i = start; i < nums.size(); i++) {
        current.push_back(nums[i]);  // Include
        backtrack(nums, i + 1, current, result);
        current.pop_back();          // Exclude (backtrack)
    }
}
```

**Memory Hook**: "ADD, RECURSE, REMOVE (backtrack)"

---

### 11. Modified Binary Search 🎯
**Memory Trick**: "Divide and Conquer with a Twist"

**Key Patterns**:

**Rotated Array**:
```cpp
// Find which half is sorted
if (nums[left] <= nums[mid]) {
    // Left half is sorted
} else {
    // Right half is sorted
}
```

**Find Peak**:
```cpp
if (nums[mid] > nums[mid + 1]) {
    // Peak is on left side (including mid)
    right = mid;
} else {
    // Peak is on right side
    left = mid + 1;
}
```

**Memory Hook**: "One side is ALWAYS sorted in rotation"

---

### 12. Bitwise XOR ⚡
**Memory Trick**: "Magic Cancellation"

**Key Properties**:
- `a ⊕ a = 0` (self-cancels)
- `a ⊕ 0 = a` (identity)  
- `a ⊕ b ⊕ a = b` (cancellation)

**Visual**:
```
3 ⊕ 5 ⊕ 3 = 5
[3 cancels out]
```

**Use Cases**:
- Find single number in pairs
- Find missing number
- Swap without temp variable

**Memory Hook**: "XOR makes duplicates DISAPPEAR!"

---

### 13. Top K Elements 🏆
**Memory Trick**: "Hall of Fame" - keep only the best K

**Two Approaches**:

**Min Heap** (for K largest):
```cpp
priority_queue<int, vector<int>, greater<int>> minHeap;
// Keep size ≤ K, smallest at top gets kicked out
```

**Max Heap** (for K smallest):
```cpp
priority_queue<int> maxHeap;
// Keep size ≤ K, largest at top gets kicked out
```

**Memory Hook**: "Keep K VIPs, kick out the WORST"

---

### 14. K-way Merge 🔀
**Memory Trick**: "Orchestra Conductor" - merge multiple sorted streams

**Visual**:
```
List1: 1→4→7
List2: 2→5→8    →  1→2→3→4→5→6→7→8→9
List3: 3→6→9
```

**Pattern**:
```cpp
priority_queue<pair<int, pair<int, int>>, 
               vector<pair<int, pair<int, int>>>, 
               greater<pair<int, pair<int, int>>>> pq;

// pq stores: {value, {listIndex, elementIndex}}
```

**Memory Hook**: "Always pick the SMALLEST from all heads"

---

### 15. 0/1 Knapsack 🎒
**Memory Trick**: "Take It or Leave It"

**Visual**:
```
Items: [💎, 📱, 💻]
Capacity: 5kg

For each item: Take or Skip?
```

**DP Pattern**:
```cpp
// dp[i][w] = max value using first i items with weight limit w
for (int i = 1; i <= n; i++) {
    for (int w = 1; w <= capacity; w++) {
        if (weight[i-1] <= w) {
            dp[i][w] = max(dp[i-1][w],                    // Don't take
                          dp[i-1][w-weight[i-1]] + value[i-1]); // Take
        } else {
            dp[i][w] = dp[i-1][w]; // Can't take (too heavy)
        }
    }
}
```

**Memory Hook**: "If it FITS, choose MAX(take, skip)"

---

### 16. Unbounded Knapsack 🔄
**Memory Trick**: "Unlimited Supply Store"

**Key Difference**: Can use same item multiple times

**Pattern**:
```cpp
for (int i = 1; i <= n; i++) {
    for (int w = weight[i-1]; w <= capacity; w++) {
        dp[w] = max(dp[w], dp[w - weight[i-1]] + value[i-1]);
    }
}
```

**Memory Hook**: "UNLIMITED items, but LIMITED capacity"

---

### 17. Fibonacci Numbers 🐰
**Memory Trick**: "Rabbit Reproduction"

**Pattern Recognition**:
- "How many ways to reach..."
- "Climbing stairs"  
- "House robber"

**Optimized Pattern**:
```cpp
int prev2 = base_case_0;
int prev1 = base_case_1;
for (int i = 2; i <= n; i++) {
    int curr = prev1 + prev2;
    prev2 = prev1;
    prev1 = curr;
}
```

**Memory Hook**: "Current = Previous + Previous Previous"

---

### 18. Palindromic Subsequence 🪞
**Memory Trick**: "Mirror Mirror"

**Visual**:
```
"racecar" = "r" + "aceca" + "r"
         mirror
```

**Expand Around Center**:
```cpp
for (int i = 0; i < n; i++) {
    expandAroundCenter(i, i);     // Odd length
    expandAroundCenter(i, i + 1); // Even length
}
```

**DP for Subsequence**:
```cpp
if (s[i] == s[j]) {
    dp[i][j] = dp[i+1][j-1] + 2; // Add both ends
} else {
    dp[i][j] = max(dp[i+1][j], dp[i][j-1]); // Skip one end
}
```

**Memory Hook**: "If MATCH, add 2; if NOT, take MAX"

---

### 19. Longest Common Subsequence 🤝
**Memory Trick**: "Finding Common Ground"

**Visual**:
```
ABCDGH
AEDFHR
Common: ADH (length 3)
```

**DP Pattern**:
```cpp
if (text1[i-1] == text2[j-1]) {
    dp[i][j] = dp[i-1][j-1] + 1; // Match: extend previous
} else {
    dp[i][j] = max(dp[i-1][j], dp[i][j-1]); // No match: take best
}
```

**Memory Hook**: "MATCH extends, NO MATCH takes BEST"

---

### 20. Topological Sort 📊
**Memory Trick**: "Prerequisite Chain"

**Visual**:
```
A → B → D
↓   ↓
C → E

Topo order: A, C, B, E, D
```

**Kahn's Algorithm**:
```cpp
1. Calculate indegrees
2. Add nodes with indegree 0 to queue
3. Process queue: remove node, decrease neighbors' indegrees
4. If processed count == total nodes: valid ordering
```

**Memory Hook**: "No PREREQUISITES first, then UNLOCK next level"

---

### 21. Union Find 🤝
**Memory Trick**: "Family Tree Merger"

**Visual**:
```
Before: 1  2    3  4
After:  1─2    3─4  (Union)
        └──────┘    (Union groups)
```

**Pattern**:
```cpp
class UnionFind {
    vector<int> parent, rank;
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        // Union by rank...
    }
};
```

**Memory Hook**: "Find BOSS, Union GROUPS"

---

### 22. Ordered Set 📚
**Memory Trick**: "Library Bookshelf" - always sorted

**Use Cases**:
- Range queries
- Closest elements
- Sliding window with order

**Pattern**:
```cpp
set<int> window;
auto it = window.lower_bound(target - k);
if (it != window.end() && *it <= target + k) {
    // Found in range
}
```

**Memory Hook**: "lower_bound finds FIRST >= target"

---

### 23. Multi-threaded 🧵
**Memory Trick**: "Traffic Light Control"

**Synchronization Tools**:
- **Mutex**: 🚦 Traffic light (one at a time)
- **Semaphore**: 🎫 Limited tickets
- **Condition Variable**: 📢 Wait for signal

**Pattern**:
```cpp
mutex m1, m2;
m1.lock(); m2.lock(); // Start locked

void first() { print(); m1.unlock(); }
void second() { m1.lock(); print(); m1.unlock(); m2.unlock(); }
void third() { m2.lock(); print(); m2.unlock(); }
```

**Memory Hook**: "LOCK gates, UNLOCK when ready"

---

### 24. Monotonic Stack 📈
**Memory Trick**: "Mountain Climber" - only goes up (or down)

**Visual**:
```
Stack: [1, 3, 5] (increasing)
New: 2 → Pop 5,3 → Push 2
Result: [1, 2]
```

**Use Cases**:
- Next Greater Element
- Largest Rectangle
- Daily Temperatures

**Pattern**:
```cpp
stack<int> st;
for (int i = 0; i < n; i++) {
    while (!st.empty() && arr[st.top()] < arr[i]) {
        int idx = st.top(); st.pop();
        result[idx] = i - idx; // Distance to next greater
    }
    st.push(i);
}
```

**Memory Hook**: "Pop SMALLER, push CURRENT"

---

### 25. Trie (Prefix Tree) 🌳
**Memory Trick**: "Word Tree" - letters as branches

**Visual**:
```
    root
   /  |  \
  c   t   w
  |   |   |
  a   h   e
  |   |   |
  t   e   b
```

**Structure**:
```cpp
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord = false;
};
```

**Memory Hook**: "Follow PATH of letters, mark END of words"

---

### 26. Hash Maps 🗺️
**Memory Trick**: "Phone Book" - instant lookup

**Common Patterns**:
- **Frequency Counter**: `map[char]++`
- **Two Sum**: `map[complement] = index`
- **Anagram Groups**: `map[sorted_string] = group`

**Memory Hook**: "O(1) lookup, but watch for COLLISIONS"

---

### 27. Intervals 📅
**Memory Trick**: "Calendar Scheduler"

**Types**:
- **Point events**: `[start, start]`
- **Range events**: `[start, end]`
- **Overlapping**: `start1 < end2 && start2 < end1`

**Memory Hook**: "If OVERLAP exists, handle CONFLICT"

---

### 28. Greedy 🏃‍♂️
**Memory Trick**: "Take Best Now, Worry Later"

**When to Use**:
- Optimal substructure
- Greedy choice property
- Local optimum → Global optimum

**Examples**:
- Jump Game: Always jump to farthest reachable
- Activity Selection: Pick earliest ending time

**Memory Hook**: "Be GREEDY locally, get OPTIMAL globally"

---

### 29. Backtracking 🔄
**Memory Trick**: "Explorer with Undo Button"

**Template**:
```cpp
void backtrack(/* state */) {
    if (/* goal reached */) {
        result.push_back(current);
        return;
    }
    
    for (/* each choice */) {
        // Make choice
        current.push_back(choice);
        
        // Recurse
        backtrack(/* new state */);
        
        // Undo choice (backtrack)
        current.pop_back();
    }
}
```

**Memory Hook**: "Try, Recurse, UNDO"

---

### 30. Dynamic Programming on Trees 🌳💰
**Memory Trick**: "House Robber in Tree Village"

**Pattern**:
```cpp
pair<int, int> dfs(TreeNode* node) {
    if (!node) return {0, 0};
    
    auto left = dfs(node->left);
    auto right = dfs(node->right);
    
    // {rob this node, skip this node}
    int rob = node->val + left.second + right.second;
    int skip = max(left.first, left.second) + 
               max(right.first, right.second);
    
    return {rob, skip};
}
```

**Memory Hook**: "ROB node, skip children OR SKIP node, take best from children"

---

## 🎨 Visual Memory Palace

### Floor 1: Arrays & Strings
- **Room 1**: 👥 Two Pointers (Meeting room)
- **Room 2**: 🪟 Sliding Window (Window shopping)
- **Room 3**: 🔄 Cyclic Sort (Musical chairs)

### Floor 2: Linked Lists & Trees
- **Room 4**: 🐢🐰 Fast/Slow Pointers (Racing track)
- **Room 5**: ↩️ LinkedList Reversal (Dance floor)
- **Room 6**: 🌊 Tree BFS (Swimming pool levels)
- **Room 7**: 🌳 Tree DFS (Hiking trails)

### Floor 3: Advanced Structures
- **Room 8**: ⚖️ Two Heaps (Balance scale room)
- **Room 9**: 🏆 Top K (Trophy room)
- **Room 10**: 📚 Ordered Set (Library)

---

## 🚀 Quick Decision Tree

```
Problem involves...
├── Array/String
│   ├── Two elements → Two Pointers
│   ├── Subarray/Substring → Sliding Window  
│   └── Range [1,n] → Cyclic Sort
├── LinkedList
│   ├── Cycle → Fast/Slow Pointers
│   └── Reverse → In-place Reversal
├── Tree
│   ├── Level by level → BFS
│   └── Root to leaf → DFS
├── Optimization
│   ├── Take/Skip → 0/1 Knapsack
│   ├── Local optimal → Greedy
│   └── All possibilities → Backtracking
└── Graph
    ├── Connectivity → Union Find
    ├── Ordering → Topological Sort
    └── Shortest path → BFS/Dijkstra
```

---

## 💡 Final Memory Tips

1. **Pattern Recognition**: Look for keywords first
2. **Visual Mapping**: Associate each pattern with a strong visual
3. **Template Practice**: Memorize the core template for each pattern
4. **Problem Mapping**: Group similar problems together
5. **Time Complexity**: Remember the "Big O" for each pattern

**The Golden Rule**: "Recognize pattern → Apply template → Optimize for edge cases"

---

*Master these patterns and you'll solve 90% of coding interview problems! 🎯*
