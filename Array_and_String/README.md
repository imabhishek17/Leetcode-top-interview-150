# 1. Merge Sorted Array ([link](https://leetcode.com/problems/merge-sorted-array/description/?envType=study-plan-v2&envId=top-interview-150))

- Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
- Output: [1,2,2,3,5,6]

- Input: nums1 = [0], m = 0, nums2 = [1], n = 1
- Output: [1]

**SOLUTION:**

```cpp

class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int len = m+n-1;
        int i = m-1, j = n-1; 
        while(i>=0 and j>=0) {
            if(nums1[i] > nums2[j])
            nums1[len--] = nums1[i--];
            else
            nums1[len--] = nums2[j--];
        }

        while(i>=0) {
            nums1[len--] = nums1[i--];
        }
        while(j>=0) {
            nums1[len--] = nums2[j--];
        }
    }
};
```


# 2. Remove Element ([link](https://leetcode.com/problems/remove-element/description/?envType=study-plan-v2&envId=top-interview-150))

Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

- Input: nums = [3,2,2,3], val = 3
- Output: 2, nums = [2,2,_,_]

- Input: nums = [0,1,2,2,3,0,4,2], val = 2
- Output: 5, nums = [0,1,4,0,3,_,_,_]

**SOLUTION:**

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int j = 0;
        for(int i=0; i<nums.size(); i++){
            if(nums[i] != val) {
                nums[j++] = nums[i];
            }
        }
        return j;
    }
};
```

# 3. Remove Duplicates from Sorted Array ([link](https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/?envType=study-plan-v2&envId=top-interview-150))

Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same. Then return the number of unique elements in nums.

Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially. The remaining elements of nums are not important as well as the size of nums.
Return k.


- Input: nums = [1,1,2]
- Output: 2, nums = [1,2,_]

- Input: nums = [0,0,1,1,1,2,2,3,3,4]
- Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]

**SOLUTION:**

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = nums.size();
        int j=1;
        for(int i=1; i<n; i++) {
            if(nums[i-1] != nums[i]) {
                nums[j++] = nums[i];
            }
        }
        return j;
    }
};
```

# 4. Remove Duplicates from Sorted Array II ([link](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/?envType=study-plan-v2&envId=top-interview-150))

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.


- Input: nums = [1,1,1,2,2,3]
- Output: 5, nums = [1,1,2,2,3,_]

- Input: nums = [0,0,1,1,1,1,2,3,3]
- Output: 7, nums = [0,0,1,1,2,3,3,_,_]


**SOLUTION:**

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = nums.size(), j=1, count=1;
        for(int i=1; i<n; i++) {
            if(nums[i-1] == nums[i]){
                count++;
            } else {
                count = 1;
            }
            if(count<=2) {
                nums[j++] = nums[i];
            }
        }
        return j;
    }
};
```

# 5. Majority Element ([link](https://leetcode.com/problems/majority-element/description/?envType=study-plan-v2&envId=top-interview-150))



**SOLUTION:**



# 6. Rotate Array ([link](https://leetcode.com/problems/rotate-array/description/?envType=study-plan-v2&envId=top-interview-150))

Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.


- Input: nums = [-1,-100,3,99], k = 2
- Output: [3,99,-1,-100]


**SOLUTION:**


```cpp
class Solution {
public:
    
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k=(k%n);
        reverse(nums, 0, n-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, n-1);
    }
};
```

# 7. Best Time to Buy and Sell Stock ([link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=top-interview-150))

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

- Input: prices = [7,1,5,3,6,4]
- Output: 5

- Input: prices = [7,6,4,3,1]
- Output: 0


**SOLUTION:**

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int mx = 0;
        int mn = prices[0];
        for(int i=1; i<prices.size(); i++){
            mn = min(mn, prices[i]);
            mx = max(mx, prices[i] - mn);
        }
        return mx;
    }
};
```

# 8. Best Time to Buy and Sell Stock II ([link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/?envType=study-plan-v2&envId=top-interview-150))


You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.


- Input: prices = [7,1,5,3,6,4]
- Output: 7

- Input: prices = [1,2,3,4,5]
- Output: 4

- Input: prices = [7,6,4,3,1]
- Output: 0


**SOLUTION:**

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
       int mx = 0;
       int n = prices.size();
       for(int i=0;i<n-1;i++){
        if(prices[i]<prices[i+1])
        mx += (prices[i+1] - prices[i]);
       }
       return mx; 
    }
};
```

# 9. Jump Game ([link](https://leetcode.com/problems/jump-game/description/?envType=study-plan-v2&envId=top-interview-150))

You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
Return true if you can reach the last index, or false otherwise.


- Input: nums = [2,3,1,1,4]
- Output: true

- Input: nums = [3,2,1,0,4]
- Output: false

**SOLUTION:**

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int maxReach = 0, n = nums.size();

        for(int i = 0 ; i < n ; i++) {
            if(i > maxReach) return false;
            maxReach = max(maxReach, i + nums[i]);
        }

        return true;
    }
};
```


# 10. Jump Game II ([link](https://leetcode.com/problems/jump-game-ii/description/?envType=study-plan-v2&envId=top-interview-150))

You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:
0 <= j <= nums[i] and
i + j < n
Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].


- Input: nums = [2,3,1,1,4]
- Output: 2

- Input: nums = [2,3,0,1,4]
- Output: 2


**SOLUTION:**

```cpp
//DP solution
class Solution {
public:

    int solve(vector<int>&nums, vector<int>&dp, int curr) {
        if(curr >= nums.size()-1) return 0;
        if(dp[curr] != -1) return dp[curr];

        int minStep = INT_MAX;
        int n = nums.size();
        int maxJump = min(n , curr + nums[curr]);
        for(int i = curr+1; i<=maxJump; ++i) {
            int cnt = solve(nums, dp, i);
            if(cnt != INT_MAX) minStep = min(minStep, cnt+1);
        }
        return  dp[curr] = minStep;
    }

    int jump(vector<int>& nums) {
      vector<int>dp (nums.size(), -1);
      return solve(nums, dp, 0);
    }
};

//Optimised Solution
class Solution {
public:
    int jump(vector<int>& nums) {
        int near = 0, far = 0, jumps = 0;

        while (far < nums.size() - 1) {
            int farthest = 0;
            for (int i = near; i <= far; i++) {
                farthest = max(farthest, i + nums[i]);
            }
            near = far + 1;
            far = farthest;
            jumps++;
        }

        return jumps;        
    }
};
```

# 11. H-Index ([link](https://leetcode.com/problems/h-index/description/?envType=study-plan-v2&envId=top-interview-150))

Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper, return the researcher's h-index.

According to the definition of h-index on Wikipedia: The h-index is defined as the maximum value of h such that the given researcher has published at least h papers that have each been cited at least h times.

- Input: citations = [3,0,6,1,5]
- Output: 3

- Input: citations = [1,3,1]
- Output: 1


**SOLUTION:**

```cpp
//By sorting
class Solution {
public:
    int hIndex(vector<int>& citations) {
        sort(citations.begin(), citations.end());
        int count = 0, n = citations.size();

        for(int i=0; i<n; i++){
            if(citations[i] >= (n-i)){
                count = max(count, (n-i));
            }
        }
        return count;
    }
};


//Optimised Solution (Bucket sort OR frequency array)
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int n = citations.size();
        vector<int>freq(n+1, 0);
        for(int i=0; i<n; i++){
            if(citations[i] <= n){
                freq[citations[i]]++;
            }
            else{
                freq[n]++;
            }
        }

        int count = 0;
        for(int i=n; i>=0; i--){
            count += freq[i];
            if(count >= i)
            return i;
        }
        return 0;
    }
};
```

# 12. Insert Delete GetRandom O(1) ([link](https://leetcode.com/problems/insert-delete-getrandom-o1/description/?envType=study-plan-v2&envId=top-interview-150))

Problem Stement : Open link


### Example

#### Input:
```plaintext
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
```

#### Output: 
```plaintext
[null, true, false, true, 2, true, false, 2]
```

**SOLUTION:**

```cpp
class RandomizedSet {
public:
    vector<int>v;
    map<int,int>m;
    RandomizedSet() {
        
    }
    
    bool insert(int val) {
        if(m.find(val)!=m.end()) return false;

        v.push_back(val);
        m.insert({val, v.size()-1});
        return true;
    }
    
    bool remove(int val) {
        if(m.find(val)==m.end()) return false;

        int idx = m[val];
        int lastElement = v.back();
        v.back() = val;
        v[idx] = lastElement;

        m[lastElement] = idx;
        v.pop_back();
        m.erase(val);
        return true;
    }
    
    int getRandom() {
        int n = v.size();
        int random = rand()%n;
        return v[random]; 
    }
};
```

# 13. Product of Array Except Self

### Example 1:

#### Input: 

```plaintext
nums = [1,2,3,4]
```

#### Output:
```plaintext
[24,12,8,6]
```

### Example 2:

#### Input: 

```plaintext
nums = [-1,1,0,-3,3]
```

#### Output: 

```plaintext
[0,0,9,0,0]
```


**SOLUTION:**

```cpp
# Brute Force (O(n²)): Using a nested loop i.e. skip the current element and multiply all other elements which is basically the product by multiplying the elements on the left side and the right side of the current index.

# Optimised but extra space
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int>res(n);    // left cummulative multiplication
        res[0] = 1; 
        for(int i = 1; i < n; i++) {
            res[i] = nums[i - 1] * res[i - 1];
        }

        vector<int>right(n);    // left cummulative multiplication
        right[n-1] = 1; 
        for(int i = n-2; i >= 0; i--) {
            right[i] = nums[i + 1] * right[i + 1];
        }

        for(int i = 0; i < n; i++) {
            res[i] = res[i] * right[i];
        }

        return res;
    }
};


# More optimized approach without taking Right Array Space (i.e. Reducing Space Complexity)
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int>res(n);    // left cummulative multiplication
        res[0] = 1; 
        for(int i = 1; i < n; i++) {
            res[i] = nums[i - 1] * res[i - 1];
        }

        int product = 1;   // handling right cummulative multiplication

        for(int i = n - 1; i >= 0; i--) {
         res[i] = product * res[i]; 
         product *= nums[i];
        }
        
        return res;
    }
};
```
# 14. Gas Station ([link](https://leetcode.com/problems/gas-station/description/?envType=study-plan-v2&envId=top-interview-150))

There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique.


### Example

#### Input:
```plaintext
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]

```

#### Output: 
```plaintext
3
```

#### Input:
```plaintext
Input: gas = [2,3,4], cost = [3,4,3]

```

#### Output: 
```plaintext
-1
```


**SOLUTION:**

```cpp
// TRICK OF THIS IS: You reach reach your financial goal only when you save more and spend less

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
      int n = gas.size();
	 int totalEarn = accumulate(begin(gas), end(gas), 0);
	 int totalSpend = accumulate(begin(cost), end(cost), 0);
	
	 if(totalEarn < totalSpend) return -1;
	
	 int result = 0;
	 int total = 0;
		
	 for(int i = 0; i < n; i++) {
	       total += gas[i] - cost[i];
		  if(total < 0) {
		   result = i + 1;
		   total = 0;
		  }
      }
	 return result;
    }
};

//TIME COMPLEXITY: O(3*N) ~ O(N)

```

# 15. Candy ([link](https://leetcode.com/problems/candy/description/?envType=study-plan-v2&envId=top-interview-150))

**SOLUTION:**

```cpp

//Brute force O(n) time and O(2*n) space

class Solution {
public:
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        vector<int> left(n, 1);
        vector<int> right(n, 1);
        int sum = 0;
        for(auto i = 1; i < n; i++) {
            if(ratings[i-1] < ratings[i]) {
                left[i] = left[i-1] + 1;
            }
        }

        for(auto i = n-2; i >= 0; i--) {
            if(ratings[i+1] < ratings[i]) {
                right[i] = right[i+1] + 1;
            }
        }

        for(auto i = 0; i < n; i++) {
            sum += max(left[i], right[i]);
        }

        return sum;
    }
};

// Optimised space in O(n) time and O(n) space

class Solution {
public:
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        vector<int> vec(n, 1);
        int sum = 0;
        for(auto i = 1; i < n; i++) {
            if(ratings[i-1] < ratings[i]) {
                vec[i] = vec[i-1] + 1;
            }
        }

        for(auto i = n-2; i >= 0; i--) {
            if(ratings[i+1] < ratings[i]) {
                vec[i] = max(vec[i], vec[i+1] + 1);
            }
        }

        for(auto i = 0; i < n; i++) {
            sum += vec[i];
        }

        return sum;
    }
};


//More space optimised i.e. in O(n) time and O(1) space

class Solution {
public:
    int candy(vector<int>& ratings) {
        
        int count = ratings.size();
        int n = ratings.size();
        int i = 1;
        while(i < n){
           if(ratings[i-1] == ratings[i]){
            i++;
            continue;
           }
           int peak = 0;
           while(i < n && ratings[i-1] < ratings[i]) {
             peak += 1;
             count += peak;
             i++;
           }
           int dip = 0;
           while(i < n && ratings[i-1] > ratings[i]) {
             dip += 1;
             count +=dip;
             i++;
           }
           count -= min(peak, dip);
        }
        return count;
    }
};


```











