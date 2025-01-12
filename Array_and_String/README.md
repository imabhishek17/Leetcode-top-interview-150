Container With Most Water (https://leetcode.com/problems/container-with-most-water/description/)

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int start = 0, end = height.size() - 1;
        int result = 0;

        while (start < end) {
            int curr_length = end - start;
            int curr_breadth = min(height[start], height[end]);
            result = max(result, curr_length * curr_breadth);

            if (height[start] < height[end]) {
                start++;
            } else {
                end--;
            }
        }
        return result;
    }
};


// in this as we are contracting length, so the only way to increase/maximise
// the area is to maximise the height (i.e. width).
```

------------------------------------------------------------------------------------------------------------------------

3-Sum (https://leetcode.com/problems/3sum/description/)

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;

        int n = nums.size();
        if (n < 3)
            return res;

        sort(nums.begin(), nums.end());
        int start = 0, end = 0;

        for (int i = 0; i < n-2; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;

            start = i + 1;
            end = n - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];

                if (sum == 0) {
                    res.push_back({nums[i], nums[start], nums[end]});
                    start++;
                    end--;
                    while (start < end && nums[start] == nums[start - 1]) {
                        start++;
                    }
                    while (start < end && nums[end] == nums[end + 1]) {
                        end--;
                    }
                } else if (sum < 0) {
                    start++;
                } else {
                    end--;
                }
            }
        }
        return res;
    }
};
```

------------------------------------------------------------------------------------------------------------------------

Triangle Numbers (https://leetcode.com/problems/valid-triangle-number/description/)

```cpp

class Solution {
public:
    int triangleNumber(std::vector<int>& nums) {
        int n = nums.size();
        int count = 0;
        sort(nums.begin(), nums.end());

        for (int i = 2; i < n; i++) { 
            int start = 0, end = i - 1;
            while (start < end) {
                if (nums[start] + nums[end] > nums[i]) {
                    count += (end - start);
                    end--;
                } else {
                    start++;
                }
            }
        }
        return count; 
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Move Zeroes (https://leetcode.com/problems/move-zeroes/description/)

```cpp

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int nz = 0;

        for (int i = 0; i < nums.size(); i++) {
            if (nums[i]) {
                nums[nz++] = nums[i];
            }
        }

        while (nz < nums.size()) {
            nums[nz++] = 0;
        }
    }
};

OR

    //with extra variable
    // class Solution {
    // public:
    //    void moveZeroes(vector<int>& nums) {
    //         int left = 0, right = 0;
    //      while(right < nums.size()) {
    //             if(nums[right] != 0) {
    //                 int temp = nums[left];
    //                 nums[left] = nums[right];
    //                 nums[right] = temp;
    //                 left++;
    //             }
    //          right++;
    //         }
    //     }
    // };


//without extra variable (space optimised)
// class Solution {
// public:
//    void moveZeroes(vector<int>& nums) {
//         int left = 0, right = 0;
// 	    while(right < nums.size()) {
//             if(nums[right] != 0) {
//                 if(left < right) {
//                     nums[left] = nums[left] + nums[right];
//                     nums[right] = nums[left] - nums[right];
//                     nums[left] = nums[left] - nums[right];
//                 }
//                 left++;           
//             }
// 	        right++;
//         }
//     }
// };

```
------------------------------------------------------------------------------------------------------------------------


Sort Colors (https://leetcode.com/problems/sort-colors/description/)

```cpp

class Solution {
public:
    void sortColors(vector<int>& nums) {
        int zero, one, two;
        int n = nums.size();
        zero = 0, one = 0, two = n - 1;
        while (one <= two) {
            if (nums[one] == 2) {
                swap(nums[one], nums[two]);
                two -= 1;
            } else if (nums[one] == 1) {
                one += 1;
            } else {
                swap(nums[one], nums[zero]);
                zero += 1;
                one += 1;
            }
        }
    }
};

// class Solution {
// public:
//    void sortColors(vector<int>& nums) {
//         int zero = 0, one = 0, two = 0;
//    int n = nums.size();
//    for(int i = 0; i < n; i++) {
//       if(nums[i] == 0) zero += 1 ;
//  if(nums[i] == 1) one += 1;
//  if(nums[i] == 2) two += 1;
//   }

//   int i = 0;
//   while(i < n){
//       while(zero > 0) {
//      nums[i++] = 0;
//      zero -= 1;
// }
// while(one > 0) {
//      nums[i++] = 1;
//      one -= 1;
// }
// while(two > 0) {
//      nums[i++] = 2;
//      two -= 1;
// }
//   }
//    }
// };

```
------------------------------------------------------------------------------------------------------------------------

Trapping Rain Water (https://leetcode.com/problems/trapping-rain-water/description/)

```cpp

class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        int left = 0, right = n - 1;
        int leftMax = height[left];
        int rightMax = height[right];
        int water = 0;

        while (left < right) {
            if (leftMax < rightMax) {
                left += 1;
                leftMax = max(height[left], leftMax);
                water += (leftMax - height[left]);
            } else {
                right -= 1;
                rightMax = max(height[right], rightMax);
                water += (rightMax - height[right]);
            }
        }
        return water;
    }
};

```

------------------------------------------------------------------------------------------------------------------------

Can Attend Meetings OR Meeting Rooms I (https://leetcode.com/problems/meeting-rooms/description/)

```cpp

//Problem statement: Write a function to check if a person can attend all the meetings scheduled without any time conflicts. Given an array intervals, where each element [s1, e1] represents a meeting starting at time s1 and ending at time e1, determine if there are any overlapping meetings. If there is no overlap between any meetings, return true; otherwise, return false.

class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {
        if (intervals.empty()) {
            return true;
        }
        sort(intervals.begin(), intervals.end());
        for (int i = 1; i < intervals.size(); ++i) {
            if (intervals[i][0] < intervals[i - 1][1]) {
                return false;
            }
        }
        return true;
    }
};

```

------------------------------------------------------------------------------------------------------------------------

Meeting Rooms II (https://www.interviewbit.com/problems/meeting-rooms/)

```cpp

/*
    Given array of time intervals, determine min # of meeting rooms required
    Ex. intervals = [[0,30],[5,10],[15,20]] -> 2

    Min heap for earliest end times, most overlap will be heap size

    Time: O(n log n)
    Space: O(n)
*/

class Solution {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
        // sort intervals by start time
        sort(intervals.begin(), intervals.end());
        
        // min heap to track min end time of merged intervals
        priority_queue<int, vector<int>, greater<int>> pq;
        pq.push(intervals[0][1]);
        
        for (int i = 1; i < intervals.size(); i++) {
            // compare curr start w/ earliest end time, if no overlap then pop
            if (intervals[i][0] >= pq.top()) {
                pq.pop();
            }
            // add new room (will replace/be same size if above was true)
            pq.push(intervals[i][1]);
        }
        
        return pq.size();
    }
};


```
------------------------------------------------------------------------------------------------------------------------

Insert Interval (https://leetcode.com/problems/insert-interval/description/)

```cpp

/*
    Given array of non-overlapping intervals & a new interval, insert & merge if necessary
    Ex. intervals = [[1,3],[6,9]], newInterval = [2,5] -> [[1,5],[6,9]]

*/

class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> res;
        int len = intervals.size();
        int len1 = newInterval.size();

       	if(len == 0 and len1 == 0) return res;
        if(len == 0 and len1 > 0) {
            res.push_back(newInterval);
            return res;
        }
        if(len > 0 and len1 == 0) return intervals;

        intervals.push_back(newInterval);

       	sort(intervals.begin(), intervals.end());
       	vector<int> temp = intervals[0];
        len = intervals.size();
        for(int i = 1; i < len; i++) {
   			if(temp[1] >= intervals[i][0]) {
      			temp[1] = max(intervals[i][1], temp[1]);
            } else {
      			res.push_back(temp);
      			temp = intervals[i];
            }
        }
        res.push_back(temp);
      	return res;
    }
};

OR

/*
    Time: O(n)
    Space: O(n)
*/
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> res;
        int n = intervals.size();

        // Add all intervals that come before the new interval
        for (int i = 0; i < n; i++) {
            if (intervals[i][1] < newInterval[0]) {
                res.push_back(intervals[i]);
            } else {
                break;
            }
        }

        /*
    Since the problem states that the intervals are sorted, there's no need to sort them again; otherwise, we would have to sort the intervals before merging.
        */
        // sort(intervals.begin(), intervals.end());  
        
        // Merge the new interval with the overlapping intervals
        int start = newInterval[0];
        int end = newInterval[1];
        for (int i = 0; i < n; i++) {
            if (intervals[i][0] <= end and intervals[i][1] >= start) {  // Since we are not sorting the intervals while insertion, we need to check both the start and end values for overlap.
                start = min(start, intervals[i][0]);
                end = max(end, intervals[i][1]);
            } else if (intervals[i][0] > end) {
                break;
            }
        }
        res.push_back({start, end});
        
        // Add all remaining intervals
        for (int i = 0; i < n; i++) {
            if (intervals[i][0] > end) {
                res.push_back(intervals[i]);
            }
        }
        
        return res;
    }
};

```
------------------------------------------------------------------------------------------------------------------------

Non-overlapping Intervals (https://leetcode.com/problems/non-overlapping-intervals/description/)

```cpp
   
/*
    Given array of intervals, return min # of intervals to remove for all non-overlapping
    Ex. intervals = [[1,2],[1,3],[2,3],[3,4]] -> 1, remove [1,3] for all non-overlapping

    Remove interval w/ longer end point, since will always overlap more or = vs shorter one

    Time: O(n log n)
    Space: O(1)
*/

class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int count = 0;
        int len = intervals.size();
        if (len <= 1)
            return count;

        sort(intervals.begin(), intervals.end());
        int prevEnd = intervals[0][1];
        for (int i = 1; i < intervals.size(); i++) {
            if (prevEnd > intervals[i][0]) {
                count++;
                prevEnd = min(prevEnd, intervals[i][1]);
            } else {
                prevEnd = intervals[i][1];
            }
        }
        return count;
    }
};

```
------------------------------------------------------------------------------------------------------------------------

Merge Intervals (https://leetcode.com/problems/merge-intervals/description/)

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

```cpp

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> res;
        int len = intervals.size();

        if (len == 0)
            return res;
        sort(intervals.begin(), intervals.end());
        vector<int> temp = intervals[0];

        for (int i = 1; i < len; i++) {
            if (temp[1] >= intervals[i][0]) {
                temp[1] = max(intervals[i][1], temp[1]);
            } else {
                res.push_back(temp);
                temp = intervals[i];
            }
        }
        res.push_back(temp);
        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Employee Free Time (https://leetcode.com/problems/employee-free-time/description/)

Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]

```cpp
class Solution {
public:
    vector<Interval> employeeFreeTime(vector<vector<Interval>> A) {
        map<int, int> m;
        for (auto &v : A) {
            for (auto &it : v) {
                m[it.start]++;
                m[it.end]--;
            }
        }
        vector<Interval> ans;
        int cnt = 0;
        for (auto it = m.begin(); it != m.end(); ++it) {
            cnt += it->second;
            if (cnt) continue;
            int start = it->first;
            ++it;
            if (it == m.end()) break;
            cnt += it->second;
            ans.emplace_back(start, it->first);
        }
        return ans;
    }
};

// Time: O(NlogT + T) where N is the total number of intervals, and T is the total number of unique times.
// Space: O(T)

```
------------------------------------------------------------------------------------------------------------------------

Valid Parentheses (https://leetcode.com/problems/valid-parentheses/description/)

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;

        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(' or s[i] == '{' or s[i] == '[') {
                st.push(s[i]);
            } else {
                if (st.empty())
                    return false;
                if ((s[i] == '}' and st.top() != '{') or
                    (s[i] == ']' and st.top() != '[') or
                    (s[i] == ')' and st.top() != '('))
                    return false;
                st.pop();
            }
        }
        return st.empty();
    }
};

/* Here above, i’ve added the if (st.empty()) return false; check before
 * accessing st.top(), to ensures that we never attempt to access st.top() when
 * the stack is empty.
 */

```
------------------------------------------------------------------------------------------------------------------------

Decode String (https://leetcode.com/problems/decode-string/description/)

Inputs:
s = "3[a2[c]]"
Output:
"accaccacc"

```cpp

class Solution {
public:
    string decodeString(string s) {
        stack<int> numStack;       // Stack for numbers
        stack<string> stringStack; // Stack for strings
        string currString = "";    // Current decoded string
        int currNumber = 0;        // Current multiplier for number

        for (char ch : s) {
            if (ch == '[') {
                // Push current string and current number onto the stacks
                stringStack.push(currString);
                numStack.push(currNumber);
                currString = ""; // Reset current string
                currNumber = 0;  // Reset current number
            } else if (ch == ']') {
                // Pop the number and previous string
                int repeatCount = numStack.top();
                numStack.pop();
                string prevString = stringStack.top();
                stringStack.pop();

                // Repeat the current string and append to the previous string
                string repeated = "";
                for (int i = 0; i < repeatCount; ++i) {
                    repeated += currString;
                }
                currString = prevString + repeated;
            } else if (isdigit(ch)) {
                // Construct the current number (may have multiple digits)
                currNumber = currNumber * 10 + (ch - '0');
            } else {
                // Append current character to the string
                currString += ch;
            }
        }

        return currString;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Longest Valid Parentheses (https://leetcode.com/problems/longest-valid-parentheses/description/)

Inputs: s = "())))"
Output: 2

Inputs: s = ""
Output: 0

```cpp

class Solution {
public:
    int longestValidParentheses(string s) {
        int left = 0, right = 0, maxLen = 0;

        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(')
                left++;
            else if (s[i] == ')')
                right++;

            if (left == right) {
                maxLen = max(maxLen, left + right);
            } else if (right > left) {
                left = 0, right = 0;
            }
        }

        left = 0, right = 0;

        for (int i = s.size() - 1; i >= 0; i--) {
            if (s[i] == '(')
                left++;
            else if (s[i] == ')')
                right++;

            if (left == right) {
                maxLen = max(maxLen, left + right);
            } else if (left > right) {
                left = 0, right = 0;
            }
        }

        return maxLen;
    }
};

// T: O(N)
// S: O(1)

    // class Solution {
    // public:
    //     int longestValidParentheses(string s) {
    //         stack<int> st;
    //         st.push(-1);     //-1 in the stack is used as a base case to
    //         correctly calculate the length of the first valid parentheses
    //         substring.

    //         int maxLen = 0;
    //         for(int i = 0; i < s.size(); i++) {
    //             if(s[i] == '('){
    //                 st.push(i);
    //             } else {
    //                 st.pop();
    //                 if(st.empty()) {
    //                     st.push(i);
    //                 } else {
    //                     maxLen = max(maxLen, i - st.top());
    //                 }
    //             }
    //         }
    //         return maxLen;
    //     }
    // };

    // T: O(N)
    // S: O(N)

```
------------------------------------------------------------------------------------------------------------------------

Linked List Cycle (https://leetcode.com/problems/linked-list-cycle/description/)

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head or !head->next) return 0;

        ListNode* slow, *fast;
        slow = head;
        fast = head;

        while(fast and fast->next) {
            slow = slow->next;
            fast = fast->next->next;

            if(slow == fast) return 1;
        }

        return 0;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Palindrome Linked List (https://leetcode.com/problems/palindrome-linked-list/description/)

```cpp
class Solution {
public:
    bool reverseLinkList(ListNode* head, ListNode*& curr) { //actual curr value update karni thi, so & pass karna pata
        if (!head)
            return 1;

        bool ans = reverseLinkList(head->next, curr);

        if (head->val != curr->val)
            return 0;
        curr = curr->next;
        return ans;
    }

    bool isPalindrome(ListNode* head) {
        if (!head or !head->next)
            return 1;
        ListNode* curr = head;
        return reverseLinkList(head, curr);
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Remove Nth Node From End of List (https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if (!head)
            return head;

        ListNode *slow, *fast;
        slow = head;
        fast = head;

        for (int i = 0; i < n; i++) {
            fast = fast->next;
        }

        if (fast == NULL) {
            ListNode* deleteHead = head->next;
            delete (head);
            return deleteHead;
        }

        while (fast and fast->next) {
            slow = slow->next;
            fast = fast->next;
        }

        ListNode* ptr = slow->next;
        slow->next = slow->next->next;
        delete (ptr);

        return head;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Reorder List (https://leetcode.com/problems/reorder-list/description/)

```cpp

class Solution {
public:
    void reverseLL(ListNode* head, ListNode*& curr) {
        if (!head)
            return;

        reverseLL(head->next, curr);

        ListNode* temp = curr->next;
        if (!curr->next) {
            return;
        } else if (curr == head) {
            curr->next = NULL;
            return;
        }
        curr->next = head;
        head->next = (temp == head) ? NULL : temp;
        curr = temp;
    }

    void reorderList(ListNode* head) {
        ListNode* curr = head;
        reverseLL(head, curr);
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Swap Nodes in Pairs (https://leetcode.com/problems/swap-nodes-in-pairs/description/)

```cpp

class Solution {
public:
    ListNode* reverseLL(ListNode* head) {
        if (!head or !head->next)
            return head;

        ListNode* temp = head->next;
        head->next = reverseLL(head->next->next);
        temp->next = head;

        return temp;
    }

    ListNode* swapPairs(ListNode* head) { return reverseLL(head); }
};
```
------------------------------------------------------------------------------------------------------------------------

Apple Harvest (Koko Eating Bananas) (https://leetcode.com/problems/koko-eating-bananas/description/)

Input: apples = [3, 6, 7], h = 8
Output: 3

```cpp

class Solution {
public:
    bool canEatBanana(vector<int>& piles, int mid, int h) {
        long long actualHours = 0;

        for (int i = 0; i < piles.size(); i++) {
            actualHours += (piles[i] / mid);

            if (piles[i] % mid != 0) {
                actualHours++;
            }
        }

        return actualHours <= h;
    }

    int minEatingSpeed(vector<int>& piles, int h) {
        int n = piles.size();

        int left = 1;
        int right = *max_element(piles.begin(), piles.end());

        while (left <= right) {
            int mid = left + ((right - left) >> 1);

            if (canEatBanana(piles, mid, h)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }
};

// Whenever we need to find the minimum optimal value as some output like here,
// consider using binary search once.

// A similar example to this problem is:
// https://leetcode.com/problems/minimum-time-to-complete-trips/description/

```
------------------------------------------------------------------------------------------------------------------------

Search in Rotated Sorted Array (https://leetcode.com/problems/search-in-rotated-sorted-array/description/)

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4 (The index of 0 in the array)

```cpp

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size();

        int left = 0;
        int right = n - 1;

        while (left <= right) {
            int mid = left + ((right - left) >> 1);

            if (nums[mid] == target)
                return mid;

            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target and target <= nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] <= target and target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Kth Largest Element in an Array (https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

```cpp

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> pq;

        if (nums.size() == 0)
            return -1;

        for (auto it : nums) {
            pq.push(it);
            if (pq.size() > k)
                pq.pop();
        }

        return pq.top();
    }
};
```
------------------------------------------------------------------------------------------------------------------------

K Closest Points to Origin (https://leetcode.com/problems/k-closest-points-to-origin/description/)

points = [[3,4],[2,2],[1,1],[0,0],[5,5]] , k = 3
Output: [[2,2],[1,1],[0,0]]

```cpp

class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        priority_queue<pair<int, pair<int, int>>> pq;

        for (auto& point : points) {
            int x = point[0];
            int y = point[1];

            int distanceFromOrigin = (x * x) + (y * y);

            pq.push({distanceFromOrigin, {x, y}});

            if (pq.size() > k) {
                pq.pop();
            }
        }

        vector<vector<int>> res;

        while (!pq.empty()) {
            res.push_back({pq.top().second.first, pq.top().second.second});
            pq.pop();
        }

        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Find K Closest Elements (https://leetcode.com/problems/find-k-closest-elements/description/)

Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]

Input: arr = [1,1,2,3,4,5], k = 4, x = -1
Output: [1,1,2,3]

```cpp

// Time Complexity: O(log(n - k)) for binary search and O(k) for slicing the
// result (overall O(log(n - k) + k), which simplifies to O(log n)). Space
// Complexity: O(1) for the in-place operations.
class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int n = arr.size();
        int left = 0, right = n - k - 1;

        while (left <= right) {
            int mid = left + ((right - left) >> 1);

            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return {arr.begin() + left, arr.begin() + left + k};
    }
};

// // Time Complexity: O(n log k) for the priority queue operations (insertions
// and deletions) and O(k log k) for sorting the result vector (overall O(n log
// k + k log k), which simplifies to O(n log k)).
// // Space Complexity: O(k) for storing the elements in the priority queue and
// result vector. class Solution { public:
//     vector<int> findClosestElements(vector<int>& arr, int k, int x) {
//         priority_queue<pair<int, int>> pq;

//         for (int i = 0; i < arr.size(); i++) {
//             pair<int, int> p = {abs(arr[i] - x), arr[i]};

//             pq.push(p);

//             if(pq.size() > k) {
//                 pq.pop();
//             }
//         }

//         vector<int> res;
//         for (int i = 0; i < k; i++) {
//             res.push_back(pq.top().second);
//             pq.pop();
//         }
//         sort(begin(res), end(res));

//         return res;
//     }
// };

// // Time Complexity: O(n log n) for sorting the pairs, and O(k log k) for
// sorting the result (overall O(n log n + k log k), which simplifies to O(n log
// n)).
// // Space Complexity: O(n) for storing the pairs and O(k) for the result
// vector. class Solution { public:
//     vector<int> findClosestElements(vector<int>& arr, int k, int x) {
//         vector<pair<int, int>> v;
//         for (int i = 0; i < arr.size(); i++) {
//             int diff = abs(arr[i] - x );
//             pair<int, int> p = {diff, arr[i]};
//             v.push_back(p);
//         }
//         sort(v.begin(), v.end());

//         vector<int> res;
//         for (int i = 0; i < k; i++) {
//             res.push_back(v[i].second);
//         }
//         sort(res.begin(), res.end());
//         return res;
//     }
// };
```
------------------------------------------------------------------------------------------------------------------------

Merge k Sorted Lists (https://leetcode.com/problems/merge-k-sorted-lists/description/)

```cpp

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {

        /*
Priority queue (min heap) banate hain, jo pair of (node value, node pointer)
rakhega. Ye ensure karega ki sabse chhoti value waala node pehle aayega.
      */
        priority_queue<pair<int, ListNode*>, vector<pair<int, ListNode*>>,
                       greater<pair<int, ListNode*>>>
            pq;

        /*
Sabhi linked lists ke nodes ko priority queue me daalte hain.
             Agar list non-null hai, to uske first node ko queue me push karte
hain.
*/
        for (auto& list : lists) {
            if (list) {
                pq.push({list->val, list}); // {value, node pointer}
            }
        }

        /*
 Dummy node ka use karenge taaki merged list ko track kar sakein.
             -1 value hai, jo sirf placeholder hai.
*/
        ListNode* dummy = new ListNode(-1);

        // Temp node banate hain, jo merged list ko traverse karega.
        ListNode* temp = dummy;

        // Jab tak priority queue khali nahi hota, tab tak merge karte rahenge.
        while (!pq.empty()) {
            // Queue ke top se sabse chhoti value wala node uthao.
            pair<int, ListNode*> p = pq.top();
            pq.pop();

            // Temp ke next pointer ko current node (p.second) se point karwa
            // do.
            temp->next = p.second;

            // Temp ko move karte hain next node pe, jisse wo merge list ke end
            // tak traverse karega.
            temp = temp->next;

            // Agar current node ke next element available hai (non-null hai),
            // to usko queue me daal do.
            if (p.second->next) {
                pq.push({p.second->next->val, p.second->next});
            }
        }

        /* Jab queue khatam ho jaye, iska matlab ki humne sabhi nodes ko merge
         kar liya hai. Dummy node se shuru hone waali merged list ko return
         karte hain, jo dummy->next ke through milti hai.
*/
        return dummy->next;
    }
};

//For this above code:
//Time Complexity: O(K log K) + O(N*K*(3*log K))where K is the number of linked lists and N is the number of nodes in each list.
//
//O(K log K) as inserting an element into the priority queue takes log K time and is repeated K times for each list head.
//
//Considering there are N nodes in each of the K linked lists, the overall number of nodes to be processed is N * K. For each of these N * K nodes:
//
//Pop: Removing the smallest element (top of the priority queue) takes log K time.
//Add: Adding the next element from the same list (when available) also takes log K time.
//Access top: Accessing the top of the priority queue for extraction or comparison also takes log K time.
//Hence, the total time complexity for the merging process across all nodes is ~ O(N * K * log K).
//
//Space Complexity : O(K) where K is the number of linked lists. The main contributor to space usage is the priority queue which holds a node from each of these lists. Regardless of the number of nodes within each list, priority queue only holds a reference to one of its nodes at a time hence the space complexity is proportional to the number of input linked lists




// class Solution {
// public:
//     // merge two linked list
//     ListNode* mergeTwoSortedLinkedList(ListNode* l1 , ListNode* l2) {
//         if(!l1) return l2;
//         if(!l2) return l1;

//         if (l1->val <= l2->val) {
//             l1 -> next = mergeTwoSortedLinkedList(l1->next, l2);
//             return l1;
//         } else {
//             l2 -> next = mergeTwoSortedLinkedList(l1, l2->next);
//             return l2;
//         }
//     }

//     //using divide and conquer
//     ListNode* partitionAndMerge(int start, int end, vector<ListNode*>& v) {
//         if(start == end) return v[start];

//         int mid = start + (end - start) /2;

//         ListNode* L1 = partitionAndMerge(start, mid, v);
//         ListNode* L2 = partitionAndMerge(mid + 1, end, v);

//         return mergeTwoSortedLinkedList(L1, L2);
//     }

//     ListNode* mergeKLists(vector<ListNode*>& lists) {
//         int k = lists.size();

//         if(k == 0) return NULL;

//         return partitionAndMerge(0, k-1, lists);
//     }
// };


//For this above code:
//Time Complexity: O( N*k(k+1)/2) ~ O(N*k^2)
//Space Complexity: O(1)
//
//Everytime two lists are merged the time complexity is proportional to the sum of the number of nodes in them as we iterate over all nodes and merge according to the data values in them.
//Assume the length of each list to be N1, N2, N3 and so on.
//
//In the first iteration, when merging the first two lists (N1 and N2), the time complexity is N1 + N2.
//In the second iteration, when merging the result of the first iteration with the third list (N3), the time complexity becomes (N1 + N2) + N3.
//In the third iteration, merging the result of the second iteration with the fourth list (N4), the time complexity becomes ((N1 + N2) + N3) + N4.
//This pattern continues until all K lists are merged.
//The total time complexity can be expressed as:
//
//T = (N1 + N2) + (N1 + N2 + N3) + .... + (N1 + N2 + N3 + .... + Nk)
//
//For simplification let's assume the length of each linked list to be proportional to N,
//
//T = N + 2N + 3N + 4N + 5N + .... + kN
//
//T = N (1 + 2 + 3 + 4 + ... + k)
//
//The sum of lengths of the lists can be calculated using the formula for the sum of the first N natural numbers:
//
//T = N (k(k+1))/2
//
//Hence, the time complexity is O( N*k(k+1)/2) ~ O(N*k^2)
//
//Space Complexity: O(1) as no additional data structures or space is allocated for storing data, only a constant space for pointers to maintain for traversing the linked list and merging them in place.

```
------------------------------------------------------------------------------------------------------------------------

Maximum Depth of a Binary Tree (https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)

```cpp

// DFS
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr)
            return 0;

        int h1 = maxDepth(root->left);
        int h2 = maxDepth(root->right);

        return 1 + max(h1, h2);
    }
};

// BFS
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root)
            return 0;

        queue<TreeNode*> q;
        q.push(root);

        int depth = 0;
        while (!q.empty()) {
            depth++;
            int n = q.size();
            while (n--) {
                TreeNode* temp = q.front();
                q.pop();
                if (temp->left)
                    q.push(temp->left);
                if (temp->right)
                    q.push(temp->right);
            }
        }
        return depth;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Path Sum (https://leetcode.com/problems/path-sum/description/)

```cpp

class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root)
            return 0;
        if (!root->left and !root->right)
            return (root->val == targetSum);

        return hasPathSum(root->left, targetSum - root->val) or
               hasPathSum(root->right, targetSum - root->val);
    }
};

//Alternate way :
     class Solution {
     public:
         bool solve(TreeNode* root, int sum, int targetSum) {
             if(!root) return false;

             sum += root->val;
             if(!root->left and !root->right){
                 return sum == targetSum;
             }

             return solve(root->left, sum, targetSum) or solve(root->right,
             sum, targetSum);
         }

         bool hasPathSum(TreeNode* root, int targetSum) {
             int sum = 0;

             return solve(root, 0, targetSum);
         }
     };
```
------------------------------------------------------------------------------------------------------------------------

Validate Binary Search Tree (https://leetcode.com/problems/validate-binary-search-tree/description/)

Input: root = [2,1,3]
Output: true

```cpp

// time - O(N), space optimised way - O(1) auxiliary space
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return isValidBSTHelper(root, LONG_MIN, LONG_MAX);
    }

    bool isValidBSTHelper(TreeNode* root, long min, long max) {
        if (!root)
            return 1;
        else if (root->val <= min or root->val >= max) {
            return 0;
        }
        return isValidBSTHelper(root->left, min, root->val) and
               isValidBSTHelper(root->right, root->val, max);
    }
};


// Time - O(N), Auxiliary space - O(N)
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if (!root)
            return 1;

        vector<int> inorder;
        helper(root, inorder);

        bool isBST = 1;

        for (int i = 1; i < inorder.size(); i++) {
            if (inorder[i] <= inorder[i - 1]) {
                isBST = 0;
            }
        }
        return isBST;
    }

    void helper(TreeNode* root, vector<int>& inorder) {
        if (!root)
            return;

        helper(root->left, inorder);
        inorder.push_back(root->val);
        helper(root->right, inorder);
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Calculate Tilt of a Binary Tree (https://leetcode.com/problems/binary-tree-tilt/description/)

Input: root = [1,2,3]
Output: 1
Explanation:
Tilt of node 2 : |0-0| = 0 (no children)
Tilt of node 3 : |0-0| = 0 (no children)
Tilt of node 1 : |2-3| = 1 (left subtree is just left child, so sum is 2; right subtree is just right child, so sum is 3)
Sum of every tilt : 0 + 0 + 1 = 1

```cpp

class Solution {
public:
    int sumTree(TreeNode* root, int& sum) {
        if (!root)
            return 0;
        if (!root->left and !root->right)
            return root->val;

        int leftSum = sumTree(root->left, sum);
        int rightSum = sumTree(root->right, sum);

        int currTilt = abs(leftSum - rightSum);
        sum += currTilt;

        return root->val + leftSum + rightSum;
    }

    int findTilt(TreeNode* root) {
        if (!root)
            return 0;
        if (!root->left and !root->right)
            return 0;

        int sum = 0;
        sumTree(root, sum);

        return sum;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Diameter of a Binary Tree (https://leetcode.com/problems/diameter-of-binary-tree/description/)

```cpp

class Solution {
public:
    int depth(TreeNode* root, int& res) {
        if (!root)
            return 0;
        if (!root->left and !root->right)
            return 1;

        int l = depth(root->left, res);
        int r = depth(root->right, res);

        res = max(res, l + r);

        return max(l, r) + 1;
    }

    int diameterOfBinaryTree(TreeNode* root) {
        if (!root)
            return 0;

        int sum = 0;
        depth(root, sum);
        return sum;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Path Sum II (https://leetcode.com/problems/path-sum-ii/description/)

Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
Explanation: There are two paths whose sum equals targetSum:
5 + 4 + 11 + 2 = 22
5 + 8 + 4 + 5 = 22

```cpp

class Solution {
public:
    void pathSumHelper(TreeNode* root, int targetSum, vector<int> currentPath, vector<vector<int>>& res) {
        if (!root)
            return;

        currentPath.push_back(root->val);

        if (!root->left and !root->right and root->val == targetSum) {
            res.push_back(currentPath);
        }

        pathSumHelper(root->left, targetSum - root->val, currentPath, res);
        pathSumHelper(root->right, targetSum - root->val, currentPath, res);
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> res;
        vector<int> currentPath;
        if (!root)
            return res;

        pathSumHelper(root, targetSum, currentPath, res);
        return res;
    }
};

OR

    //( here below this is same as above added &(pass by reference for
    // currentPath) and that’s why did res.pop_back(); )

class Solution {
public:
    void pathSumHelper(TreeNode* root, int targetSum, vector<int>& currentPath, vector<vector<int>>& res) {
        if (!root)
            return;

        currentPath.push_back(root->val);

        if (!root->left and !root->right and root->val == targetSum) {
            res.push_back(currentPath);
        }

        pathSumHelper(root->left, targetSum - root->val, currentPath, res);
        pathSumHelper(root->right, targetSum - root->val, currentPath, res);

        currentPath.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> res;
        vector<int> currentPath;
        if (!root)
            return res;

        pathSumHelper(root, targetSum, currentPath, res);
        return res;
    }
};

/*Quick Note on Passing by Value vs. Reference + Backtracking
Passing by Value (vector<int> currentPath):
Each recursive call gets its own copy of currentPath.
No need for explicit backtracking (pop_back) since changes are local to the call.
Passing by Reference (vector<int>& currentPath):
All calls share the same path — changes are reflected across all calls.
Backtracking is crucial: After modifying the path (push_back), you must undo the change (pop_back) before returning from the recursive call to avoid mutating the path for other calls.
Key Takeaway:
If passing by reference, always pop_back after recursive calls to backtrack and maintain the correct path for other calls.*/

```
------------------------------------------------------------------------------------------------------------------------

Longest Univalue Path (https://leetcode.com/problems/longest-univalue-path/description/)

Statement: Return the length of the longest path, where each node in the path has the same value. This path may or may not pass through the root.
Input: root = [5,4,5,1,1,null,5]
Output: 2
Explanation: The shown image shows that the longest path of the same value (i.e. 5).

```cpp

class Solution {
public:
    int solve(TreeNode* root, int& maxLen) {
        if (!root)
            return 0;

        int leftLen = solve(root->left, maxLen);
        int rightLen = solve(root->right, maxLen);

        if (!root->left or root->left->val != root->val)
            leftLen = 0;
        if (!root->right or root->right->val != root->val)
            rightLen = 0;

        maxLen = max(maxLen, leftLen + rightLen);

        return max(leftLen, rightLen) + 1;
    }

    int longestUnivaluePath(TreeNode* root) {
        int maxLen = 0;
        solve(root, maxLen);
        return maxLen;
    }
};


// Use len = 0 instead of return 0 to continue recursion and avoid premature
// termination. This allows you to update path lengths while still exploring
// both subtrees and properly updating global values (like maxLen).

Quick Recap for above code:
Don't prematurely return 0 for leaf nodes. (i.e. like  if(!root -> left and !root -> right) return 0;) Handle them the same way as any other node, checking if their value matches the parent's value to extend the path.
Extend the univalue path if the child matches the node's value (either left or right).
Stop recursion only for null nodes, not for leaf nodes.

```
------------------------------------------------------------------------------------------------------------------------

Level Order Sum (https://leetcode.com/problems/level-order-traversal/description/)

```cpp

class Solution {
public:
    vector<long long> levelOrderSum(TreeNode* root) {
        // If the tree is empty, return an empty vector.
        if (!root)
            return {};

        vector<long long> result; // To store the sum of each level
        queue<TreeNode*> que;     // Queue to help with level order traversal
        que.push(root);

        while (!que.empty()) {
            long long levelSum = 0;     // Sum of nodes at the current level
            int levelSize = que.size(); // Number of nodes at the current level

            // Traverse all nodes at the current level
            for (int i = 0; i < levelSize; ++i) {
                TreeNode* node = que.front();
                que.pop();

                // Add node value to level sum
                levelSum += node->val;

                // Enqueue left and right children if they exist
                if (node->left)
                    que.push(node->left);
                if (node->right)
                    que.push(node->right);
            }

            // Store the sum of the current level
            result.push_back(levelSum);
        }

        return result;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Kth Largest Sum in a Binary Tree (https://leetcode.com/problems/kth-largest-sum-in-a-binary-tree/)

```cpp

class Solution {
public:
    long long kthLargestLevelSum(TreeNode* root, int k) {
        priority_queue<long long, vector<long long>, greater<long>> pq;

        queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            long long n = que.size();
            long long sum = 0;

            while (n--) {
                TreeNode* temp = que.front();
                que.pop();
                sum += temp->val;

                if (temp->left)
                    que.push(temp->left);
                if (temp->right)
                    que.push(temp->right);
            }
            pq.push(sum);
            if (pq.size() > k) {
                pq.pop();
            }
        }
        if (pq.size() < k)
            return -1;
        return pq.top();
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Binary Tree Right Side View (https://leetcode.com/problems/binary-tree-right-side-view/description/)

```cpp

class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if (!root)
            return res;

        queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            int rightMost;
            int levelSize = que.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode* temp = que.front();
                que.pop();
                if (i == levelSize - 1)
                    rightMost = temp->val;

                if (temp->left)
                    que.push(temp->left);
                if (temp->right)
                    que.push(temp->right);
            }
            res.push_back(rightMost);
        }
        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Zigzag Level Order Traversal (https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/)

```cpp

class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root)
            return res;

        queue<TreeNode*> que;

        que.push(root);
        bool isEvenLevel = true;

        while (!que.empty()) {
            int levelSize = que.size();
            vector<int> v(levelSize);

            for (int i = 0; i < levelSize; i++) {
                TreeNode* temp = que.front();
                que.pop();

                if (temp->left)
                    que.push(temp->left);
                if (temp->right)
                    que.push(temp->right);

                int idx = 0;
                if (isEvenLevel) {
                    idx = i;
                } else {
                    idx = levelSize - 1 - i;
                }
                v[idx] = temp->val;
            }
            res.push_back(v);
            isEvenLevel = !isEvenLevel;
        }
        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Maximum Width of Binary Tree (https://leetcode.com/problems/maximum-width-of-binary-tree/description/)

```cpp

class Solution {
public:

    typedef unsigned long long ll;

    int widthOfBinaryTree(TreeNode* root) {
        if(!root) return 0;

        queue<pair<TreeNode*, ll>> que;
        que.push({root, 0});

        ll maxWidth = 0;

        while(!que.empty()) {
            int levelSize = que.size();

            ll leftMost = que.front().second;
            ll rightMost = que.back().second;
            maxWidth = max(maxWidth, rightMost - leftMost + 1);

            for(int i = 0; i < levelSize; i++) {
                TreeNode* temp = que.front().first;
                ll idx = que.front().second;

                que.pop();

                if(temp->left) que.push({temp->left, 2 * idx + 1});
                if(temp->right) que.push({temp->right, 2 * idx + 2});
            }
        }

        return maxWidth;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Best Time to Buy and Sell Stock (https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)

```cpp

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (prices.size() == 0) {
            return 0;
        }

        int mn = prices[0];
        int mx = INT_MIN;

        for (int i = 0; i < n; i++) {
            mn = min(mn, prices[i]);
            mx = max(mx, prices[i] - mn);
        }
        return mx;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Gas Station (https://leetcode.com/problems/gas-station/description/)

```cpp

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int gasSum = 0;
        int costSum = 0;

        for (int i = 0; i < gas.size(); i++) {
            gasSum += gas[i];
        }

        for (int i = 0; i < gas.size(); i++) {
            costSum += cost[i];
        }

        if (gasSum < costSum)
            return -1;

        int res = 0, total = 0;

        for (int i = 0; i < gas.size(); i++) {
            total += (gas[i] - cost[i]);
            if (total < 0) {
                total = 0;
                res = i + 1;
            }
        }

        return res;
    }
};

// In this above problem, the key idea is to find the station where the amount
// of gas you can fill is greater than or equal to the cost required to reach
// the next station from the current one.

------------------------------------------------------------------------------------------------------------------------

Jump Game (https://leetcode.com/problems/jump-game/description/)

Input: nums = [1, 3, 0, 1, 4]
Output: true


// Best approach with O(N) and O(1) time and space respectively.
 class Solution {
 public:
     bool canJump(vector<int>& nums) {
         int maxJump = 0;

         for(int i = 0; i < nums.size(); i++) {
             if(i > maxJump) return false;
             maxJump = max(maxJump, i + nums[i]);
         }
         return true;
     }
 };


//class Solution {
//public:
//    // bottom-up approach : O(N*N) time complexity, O(N) space complexity
//    bool canJump(vector<int>& nums) {
//        int n = nums.size();
//
//        vector<int> dp(n, 0);
//
//        dp[0] = true;
//
//        for (int i = 1; i < n; i++) {
//            for (int j = i - 1; j >= 0; j--) {
//                if (j + nums[j] >= i and dp[j] == true) {
//                    dp[i] = true;
//                    break;
//                }
//            }
//        }
//
//        return dp[n - 1];
//    }
//};

// top-down approach : O(N*N) time complexity, O(N) space complexity
// class Solution {
// public:
//     bool solve(vector<int>& nums, int n, int idx, int dp[]) {
//         if(idx >= n-1) return true;
//
//         if(dp[idx] != -1) return dp[idx];
//
//         for(int i = 1; i <= nums[idx]; i++) {
//             if(solve(nums, n, idx+i, dp))
//                 return dp[idx] = true;
//         }
//         return dp[idx] = false;
//     }
//
//     bool canJump(vector<int>& nums) {
//         int n = nums.size();
//         int dp[n];
//         memset(dp, -1, sizeof(dp));
//         return solve(nums, n, 0, dp);
//     }
// };

```
------------------------------------------------------------------------------------------------------------------------

Count Vowels in Substrings (https://www.hellointerview.com/learn/code/prefix-sum/count-vowels)

Input: word = "prefixsum"
queries = [[0, 2], [1, 4], [3, 5]]

Output: [1, 2, 1]

```cpp

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> countOfVowelsInSubstring(string s, vector<vector<int>>& queries) {
        int n = s.size();

        // Initialize the result vector
        vector<int> res;

        // Edge case: If the string is empty, return an empty result
        if (n == 0)
            return res;

        // Initialize prefix sum array to store the count of vowels up to index
        // i
        vector<int> prefix(n + 1, 0);

        // We can use a simple array to check if a character is a vowel.
        // Array size is 26 (for 'a' to 'z'). Vowel characters will be marked
        // as 1.
        bool isVowel[26] = {0};
        isVowel['a' - 'a'] = 1; // 'a'
        isVowel['e' - 'a'] = 1; // 'e'
        isVowel['i' - 'a'] = 1; // 'i'
        isVowel['o' - 'a'] = 1; // 'o'
        isVowel['u' - 'a'] = 1; // 'u'

        // Precompute prefix sum for vowels (start from i = 0)
        for (int i = 0; i < n; i++) {
            prefix[i + 1] =
                prefix[i] +
                isVowel[s[i] - 'a']; // Check if the character is a vowel
        }

        // Process each query
        for (const auto& query : queries) {
            int left = query[0];
            int right = query[1];

            // The vowel count in the substring s[left:right+1] is:
            res.push_back(prefix[right + 1] - prefix[left]);
        }

        return res;
    }
};

int main() {
    Solution sol;
    string s = "prefixsum";
    vector<vector<int>> queries = {{0, 2}, {1, 4}, {3, 5}};

    // Get the result from the function
    vector<int> result = sol.countOfVowelsInSubstring(s, queries);

    // Print the results
    for (int count : result) {
        cout << count << " ";
    }
    cout << endl;

    return 0;
}
```
------------------------------------------------------------------------------------------------------------------------

Subarray Sum Equals K (https://leetcode.com/problems/subarray-sum-equals-k/description/)

Input: nums = [3, 4, 7, 2, -3, 1, 4, 2] , k = 7
Output: 4

```cpp

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();

        int currSum = 0, count = 0;
        map<int, int> mp;

        for (int i = 0; i < n; i++) {
            currSum += nums[i];

            if (currSum == k)
                count++;

            if (mp.find(currSum - k) != mp.end())
                count += mp[currSum - k];

            mp[currSum]++;
        }

        return count;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Spiral Matrix (https://leetcode.com/problems/spiral-matrix/description/)

```cpp

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int n = matrix.size();
        int m = matrix[0].size();

        vector<int> res;

        int dir = 0;
        int top = 0, down = n - 1;
        int left = 0, right = m - 1;

        while (top <= down and left <= right) {
            if (dir == 0) {
                for (int i = left; i <= right; i++) {
                    res.push_back(matrix[top][i]);
                }
                top++;
            }
            if (dir == 1) {
                for (int i = top; i <= down; i++) {
                    res.push_back(matrix[i][right]);
                }
                right--;
            }
            if (dir == 2) {
                for (int i = right; i >= left; i--) {
                    res.push_back(matrix[down][i]);
                }
                down--;
            }
            if (dir == 3) {
                for (int i = down; i >= top; i--) {
                    res.push_back(matrix[i][left]);
                }
                left++;
            }
            dir++;
            if (dir == 4) {
                dir = 0;
            }
        }
        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Rotate Image (https://leetcode.com/problems/rotate-image/description/)

```cpp

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        // transpose and reverse

        int rowSize = matrix[0].size(), colSize = matrix.size();

        for (int row = 0; row < rowSize; row++) {
            for (int col = row; col < colSize; col++) {
                swap(matrix[row][col], matrix[col][row]);
            }
        }

        for (int row = 0; row < rowSize; row++) {
            reverse(matrix[row].begin(), matrix[row].end());
        }
    }
};

// class Solution {
// public:

//     void reverse1(vector<int>& v){
//         int n = v.size()-1;
//         int i = 0, j = n;

//         while(i < j) {
//             v[i] = v[i] ^ v[j];
//             v[j] = v[i] ^ v[j];
//             v[i] = v[i] ^ v[j];
//             i++; j--;
//         }
//     }

//     void rotate(vector<vector<int>>& matrix) {
//         // transpose and reverse

//         int rowSize = matrix[0].size(), colSize = matrix.size();

//         for(int row = 0; row < rowSize; row++) {
//             for(int col = row; col < colSize; col++) {
//                 if(row < col){
//                     matrix[row][col] = matrix[row][col] ^ matrix[col][row];
//                     matrix[col][row] = matrix[row][col] ^ matrix[col][row];
//                     matrix[row][col] = matrix[row][col] ^ matrix[col][row];
//                 }
//             }
//         }

//         for(int row = 0; row < rowSize; row++) {
//             reverse1(matrix[row]);
//         }
//     }
// };

// // 1 2 3
// // 4 5 6
// // 7 8 9

// transpose
//  // 1 4 7
//  // 2 5 8
//  // 3 6 9

// reverse
//  // 7 4 1
//  // 8 5 2
//  // 9 6 3

General Tip: If the swap function is not allowed in an interview and you are required to use bitwise swapping or arithmetic swapping, always ensure to include a condition like if (row < col) (or similar). This is important because, without such a check, when i == j (in the case of odd-sized matrices or lists), the same element could be swapped with itself, potentially leading to unwanted results such as setting the element to 0 (for integers) or null (for characters/strings) in certain scenarios.

```
------------------------------------------------------------------------------------------------------------------------

Set Matrix Zeroes (https://leetcode.com/problems/set-matrix-zeroes/description/)

Input:
matrix = [
    [0,2,3],
    [4,5,6],
    [7,8,9]
]

Output:
[
    [0,0,0],
    [0,5,6],
    [0,8,9]
]

```cpp

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        bool firstRow = 0,
             firstCol = 0; // for check if first row or column contains zero
        int rowSize = matrix.size(), colSize = matrix[0].size();

        // set zero(i.e. markers) in first row and col
        for (int row = 0; row < rowSize; row++) {
            for (int col = 0; col < colSize; col++) {
                if (matrix[row][col] == 0) {
                    if (row == 0)
                        firstRow = 1;
                    if (col == 0)
                        firstCol = 1;

                    matrix[row][0] = 0;
                    matrix[0][col] = 0;
                }
            }
        }

        // evaluate for submatrix (i.e. apart from first row and first column)
        for (int row = 1; row < rowSize; row++) {
            for (int col = 1; col < colSize; col++) {
                if (matrix[0][col] == 0 or matrix[row][0] == 0) {
                    matrix[row][col] = 0;
                }
            }
        }

        if (firstRow) {
            for (int col = 0; col < colSize; col++) {
                matrix[0][col] = 0;
            }
        }

        if (firstCol) {
            for (int row = 0; row < rowSize; row++) {
                matrix[row][0] = 0;
            }
        }
    }
};

// T: O(2*N*M)
// S: O(1)

```
------------------------------------------------------------------------------------------------------------------------

Find if Path Exists in Graph (https://leetcode.com/problems/find-if-path-exists-in-graph/)

Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
Output: true
Explanation: There are two paths from vertex 0 to vertex 2:
- 0 → 1 → 2
- 0 → 2

Input: n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5
Output: false
Explanation: There is no path from vertex 0 to vertex 5.


```cpp

// adjacing matrix and dfs implementation (will be getting TLE or Memory limit exceeded)
class Solution {
public:
    // DFS helper function
    bool dfs(vector<vector<int>>& graph, int curr, int destination,
             vector<bool>& vis, int n) {
        // If we reach the destination, return true
        if (curr == destination)
            return 1;

        // Mark the current node as visited
        vis[curr] = 1;

        // Explore all possible neighbors (nodes) of the current node
        for (int i = 0; i < n; i++) {
            if (!vis[i] and
                graph[curr][i] ==
                    1) { // If there's an edge and the node is not visited
                if (dfs(graph, i, destination, vis, n)) {
                    return 1; // If we find a valid path, return true
                }
            }
        }

        return 0; // No valid path found from this node
    }

    bool validPath(int n, vector<vector<int>>& edges, int source,
                   int destination) {
        vector<vector<int>> graph(n, vector<int>(n, 0));

        // Populate the adjacency matrix with edges
        for (vector<int> edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph[u][v] = 1; // Edge from u to v
            graph[v][u] = 1; // Edge from v to u (undirected graph)
        }

        // Vector to keep track of visited nodes
        vector<bool> vis(n, 0);

        // Start DFS from the source
        return dfs(graph, source, destination, vis, n);
    }
};

____________________________________________________________________

// adjacing list and dfs implementation
class Solution {
public:
    bool dfs(vector<vector<int>>& graph, int curr, int destination,
             vector<bool>& vis) {
        if (curr == destination)
            return 1;

        vis[curr] = 1;

        for (auto neighbor : graph[curr]) {
            // always check visited first, then if not visited then apply
            // recursion on child
            if (!vis[neighbor]) {
                if (dfs(graph, neighbor, destination, vis)) {
                    return 1;
                }
            }
        }

        return 0;
    }

    bool validPath(int n, vector<vector<int>>& edges, int source,
                   int destination) {
        vector<vector<int>> graph(n);

        for (vector<int> edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph[u].push_back(v);
            graph[v].push_back(u);
        }

        vector<bool> vis(n, 0);

        return dfs(graph, source, destination, vis);
    }
};

____________________________________________________________________

// adjacing matrix and bfs implementation (will be getting TLE or Memory limit
// exceeded)
class Solution {
public:
    bool validPath(int n, std::vector<std::vector<int>>& edges, int source,
                   int destination) {
        // Step 1: Create the adjacency matrix
        vector<vector<int>> graph(n, vector<int>(n, 0));

        // Fill the adjacency matrix
        for (vector<int> edge : edges) {
            int u = edge[0], v = edge[1];
            graph[u][v] = 1;
            graph[v][u] = 1;
        }

        // Step 2: Early exit if source == destination
        if (source == destination)
            return true;

        // Step 3: BFS initialization
        vector<bool> visited(n, false);
        queue<int> q;
        q.push(source);
        visited[source] = true;

        // Step 4: BFS loop
        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            if (curr == destination)
                return true; // Early exit if destination is found

            // Check all possible neighbors of the current node
            for (int i = 0; i < n; ++i) {
                if (!visited[i] and
                    graph[curr][i] ==
                        1) { // there is an edge and the node is not visited
                    visited[i] = true;
                    q.push(i);
                }
            }
        }

        // Step 5: Return false if no path is found
        return false;
    }
};

____________________________________________________________

// adjacing list and bfs implementation
class Solution {
public:
    bool validPath(int n, std::vector<std::vector<int>>& edges, int source,
                   int destination) {
        vector<vector<int>> graph(n);

        for (vector<int> edge : edges) {
            int u = edge[0], v = edge[1];
            graph[u].push_back(v);
            graph[v].push_back(u);
        }

        vector<bool> vis(n, false);
        queue<int> q;
        q.push(source);
        vis[source] = true;

        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            if (curr == destination)
                return true;

            for (int neighbor : graph[curr]) {
                if (!vis[neighbor]) {
                    q.push(neighbor);
                    vis[neighbor] = 1;
                }
            }
        }

        return false;
    }
};

```
------------------------------------------------------------------------------------------------------------------------

Numbers With Same Consecutive Differences (https://leetcode.com/problems/numbers-with-same-consecutive-differences/)

Input: n = 3, k = 7
Output: [181,292,707,818,929]
Explanation: Note that 070 is not a valid number, because it has leading zeroes.

```cpp

// dfs
class Solution {
public:
    void dfs(int num, int n, int k, vector<int>& res) {
        if (n == 0) {
            res.push_back(num);
            return;
        }

        int lastDigit = num % 10;

        if (lastDigit + k <= 9)
            dfs(num * 10 + lastDigit + k, n - 1, k, res);
        if (k != 0 and lastDigit - k >= 0)
            dfs(num * 10 + lastDigit - k, n - 1, k, res);
    }

    vector<int> numsSameConsecDiff(int n, int k) {
        vector<int> res;

        for (int i = 1; i <= 9; i++) {
            dfs(i, n - 1, k, res);
        }

        return res;
    }
};

// bfs
class Solution {
public:
    vector<int> numsSameConsecDiff(int n, int k) {
        vector<int> res;

        // Edge case: if n == 1, we can directly return digits 0-9.
        if (n == 1) {
            for (int i = 0; i <= 9; ++i) {
                res.push_back(i);
            }
            return res;
        }

        // Initialize a queue for BFS. Each element is a pair (number, current
        // length)
        queue<pair<int, int>> q;

        // Start BFS from each number between 1 to 9 (to ensure the number
        // doesn't start with 0)
        for (int i = 1; i <= 9; ++i) {
            q.push(
                {i, 1}); // Start with the number and its length (initially 1)
        }

        // Perform BFS to generate all numbers with the required properties
        while (!q.empty()) {
            auto [num, len] = q.front();
            q.pop();

            // If the current number has reached the required length, add it to
            // result
            if (len == n) {
                res.push_back(num);
                continue; // Skip further exploration for this number
            }

            // Get the last digit of the current number
            int lastDigit = num % 10;

            // Try the two possible next digits
            if (lastDigit + k <= 9) {
                q.push({num * 10 + (lastDigit + k),
                        len + 1}); // Add k to the last digit
            }
            if (k != 0 && lastDigit - k >= 0) {
                q.push({num * 10 + (lastDigit - k),
                        len + 1}); // Subtract k from the last digit
            }
        }

        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Number of Provinces (https://leetcode.com/problems/number-of-provinces/)

Statement: You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2

```cpp

 // bfs
 class Solution {
 public:
     int findCircleNum(vector<vector<int>>& isConnected) {
         int n = isConnected.size();

         int cc_count = 0;
         vector<bool> vis(n);

         for (int i = 0; i < n; i++) {
             if(!vis[i]) {
                 cc_count++;
                 queue<int> q;
                 q.push(i);
                 vis[i] = 1;

             // Perform BFS to explore all cities in the same province
                 while(!q.empty()) {
                     int levelSize = q.size();
                     for(int sz = 0; sz < levelSize; sz++) {
                         int curr = q.front();
                         q.pop();
                 // If city 'k' is connected to 'curr' and hasn't been visited, visit it
                         for(int k = 0; k < n; k++) {
                             if(!vis[k] and isConnected[curr][k]) {
                                 q.push(k);
                                 vis[k] = 1;
                             }
                         }
                     }
                 }
             }
         }

         return cc_count;
     }
 };


 // dfs
  class Solution {
  public:
      void dfs(vector<vector<int>>& isConnected, int city, vector<bool>& vis) {
          vis[city] = 1;

          for(int i = 0; i < isConnected.size(); i++) {
              if(!vis[i] and isConnected[city][i]) {
                  dfs(isConnected, i, vis);
              }
          }
      }

      int findCircleNum(vector<vector<int>>& isConnected) {
          int n = isConnected.size();

          int cc_count = 0;
          vector<bool> vis(n);

          for (int i = 0; i < n; i++) {
              if(!vis[i]) {
                  cc_count++;
                  dfs(isConnected, i, vis);
              }
          }

          return cc_count;
      }
  };

```

------------------------------------------------------------------------------------------------------------------------

Number of Islands (https://leetcode.com/problems/number-of-islands/)

Statement: Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
You may assume all four edges of the grid are all surrounded by water.

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3


```cpp

//BFS Solution
class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    int numIslands(vector<vector<char>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        int cc_count = 0;

        vector<vector<int>> vis(row, vector<int>(col, 0));

        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                if(!vis[i][j] and grid[i][j] == '1') {
                    cc_count++;

                    queue<pair<int, int>>q;
                    q.push({i, j});

                    vis[i][j] = 1;

                    while(!q.empty()) {
                        int levelSize = q.size();
                        for(int sz = 0; sz < levelSize; sz++) {
                            auto [currRow, currCol] = q.front();
                            q.pop();

                            vector<vector<int>> direc = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};

                            for (int dir = 0; dir < 4; dir++) {
                                int newRow = currRow + direc[dir][0];
                                int newCol = currCol + direc[dir][1];

                                if(isSafe(newRow, newCol, row, col) and !vis[newRow][newCol] and grid[newRow][newCol] == '1') {
                    q.push({newRow, newCol});
                    vis[newRow][newCol] = 1;
                }
                            }
                        }
                    }
                }
            }
        }

        return cc_count;
    }
};
_____________________________________________

//DFS Solution
 class Solution {
 public:
     bool isSafe(int newRow, int newCol, int row, int col) {
         int checkRow = (newRow >= 0 and newRow < row);
         int checkCol = (newCol >= 0 and newCol < col);

         return checkRow and checkCol;
     }

     void dfs(vector<vector<char>>& grid, vector<vector<int>>& vis, int i, int j, int row, int col) {
         vis[i][j] = 1;

         vector<vector<int>> direc = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};

         for (int dir = 0; dir < 4; dir++) {
             int newRow = i + direc[dir][0];
             int newCol = j + direc[dir][1];

             if(isSafe(newRow, newCol, row, col) and !vis[newRow][newCol] and grid[newRow][newCol] == '1') {
                 dfs(grid, vis, newRow, newCol, row, col);
             }
         }
     }

     int numIslands(vector<vector<char>>& grid) {
         int row = grid.size();
         int col = grid[0].size();

         int cc_count = 0;

         vector<vector<int>> vis(row, vector<int>(col, 0));

         for(int i = 0; i < row; i++) {
             for(int j = 0; j < col; j++) {
                 if(!vis[i][j] and grid[i][j] == '1') {
                     cc_count++;
                     dfs(grid, vis, i, j, row, col);
                 }
             }
         }

         return cc_count;
     }
 };

```
------------------------------------------------------------------------------------------------------------------------

Number of Closed Islands (https://leetcode.com/problems/number-of-closed-islands/description/)

Given a 2D grid consists of 0s (land) and 1s (water).
An island is a maximal 4-directionally connected group of 0s and a closed island is an island totally (all left, top, right, bottom) surrounded by 1s.
Return the number of closed islands.

Input: grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
Output: 2
Explanation:
Islands in gray are closed because they are completely surrounded by water (group of 1s).


```cpp

class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    void dfs(vector<vector<int>>& grid, int i, int j, int row, int col) {
        grid[i][j] = 1; // Mark the current cell as visited by changing 0 to 1
                        // (or any value other than 0)

        vector<vector<int>> direc = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};

        for (int dir = 0; dir < 4; dir++) {
            int newRow = i + direc[dir][0];
            int newCol = j + direc[dir][1];

            if (isSafe(newRow, newCol, row, col) and grid[newRow][newCol] == 0)
                dfs(grid, newRow, newCol, row, col);
        }
    }

    int closedIsland(vector<vector<int>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        int cc_count = 0;

        // Perform DFS from all boundary cells (edges of the grid)
        // to mark all land connected to the boundary as visited
        // This ensures that boundary-connected land won't be counted as closed
        // islands
        for (int i = 0; i < row; i++) {
            if (grid[i][0] == 0)
                dfs(grid, i, 0, row, col); // Check left boundary
            if (grid[i][col - 1] == 0)
                dfs(grid, i, col - 1, row, col); // Check right boundary
        }

        for (int i = 0; i < col; i++) {
            if (grid[0][i] == 0)
                dfs(grid, 0, i, row, col); // Check top boundary
            if (grid[row - 1][i] == 0)
                dfs(grid, row - 1, i, row, col); // Check bottom boundary
        }

        // Now check for the closed islands in the inner grid
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                // If the cell is land (0), perform DFS to mark the entire
                // island
                if (grid[i][j] == 0) {
                    cc_count++; // Increment count for each new closed island
                    dfs(grid, i, j, row, col); // Perform DFS to mark the entire
                                               // island as visited
                }
            }
        }

        return cc_count;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Flood Fill (https://leetcode.com/problems/flood-fill/description/)

Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]


```cpp

// BFS Solution
class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc,
                                  int color) {
        int original_color = image[sr][sc];

        if (original_color == color)
            return image;

        int row = image.size();
        int col = image[0].size();

        queue<pair<int, int>> q;
        q.push({sr, sc});

        image[sr][sc] = color;

        while (!q.empty()) {
            int levelSize = q.size();
            for (int sz = 0; sz < levelSize; sz++) {
                auto [currRow, currCol] = q.front();
                q.pop();

                image[currRow][currCol] = color;

                vector<vector<int>> direc = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};

                for (int dir = 0; dir < 4; dir++) {
                    int newRow = currRow + direc[dir][0];
                    int newCol = currCol + direc[dir][1];

                    if (isSafe(newRow, newCol, row, col) and
                        image[newRow][newCol] == original_color) {
                        q.push({newRow, newCol});
                        image[newRow][newCol] = color;
                    }
                }
            }
        }

        return image;
    }
};

_____________________________________________

// DFS Solution
class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    void dfs(vector<vector<int>>& image, int original_color, int color, int sr,
             int sc, int row, int col) {
        image[sr][sc] = color;

        vector<vector<int>> direc = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};

        for (int dir = 0; dir < 4; dir++) {
            int newRow = sr + direc[dir][0];
            int newCol = sc + direc[dir][1];

            if (isSafe(newRow, newCol, row, col) and
                image[newRow][newCol] == original_color)
                dfs(image, original_color, color, newRow, newCol, row, col);
        }
    }

    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc,
                                  int color) {
        int original_color = image[sr][sc];

        if (original_color == color)
            return image;

        int row = image.size();
        int col = image[0].size();

        dfs(image, original_color, color, sr, sc, row, col);

        return image;
    }
};

```

------------------------------------------------------------------------------------------------------------------------

Surrounded Regions (https://leetcode.com/problems/surrounded-regions/description/)

To capture a surrounded region, replace all 'O's with 'X's in-place within the original board. You do not need to return anything.
Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]


```cpp

class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    void dfs(vector<vector<char>>& board, int row, int col, int currRow, int currCol) {
        board[currRow][currCol] = 'A';

        vector<vector<int>> direc = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};

        for(int dir = 0; dir < 4; dir++) {
            int newRow = currRow + direc[dir][0];
            int newCol = currCol + direc[dir][1];

            if(isSafe(newRow, newCol, row, col) and board[newRow][newCol] == 'O') {
                dfs(board, row, col, newRow, newCol);
            }
        }
    }


    void solve(vector<vector<char>>& board) {
        int row = board.size();
        int col = board[0].size();

        for(int i = 0; i < row; i++) {
            if(board[i][0] == 'O') dfs(board, row, col, i, 0);
            if(board[i][col - 1] == 'O') dfs(board, row, col, i, col - 1);
        }

        for(int j = 0; j < col; j++) {
            if(board[0][j] == 'O') dfs(board, row, col, 0, j);
            if(board[row - 1][j] == 'O') dfs(board, row, col, row - 1, j);
        }

        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                if(board[i][j] != 'A') {
                    board[i][j] = 'X';
                } else {
                    board[i][j] = 'O';
                }
            }
        }
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Pacific Atlantic Water Flow (https://leetcode.com/problems/pacific-atlantic-water-flow/description/)

```cpp

class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    void dfs(vector<vector<int>>& heights, int row, int col, int currRow, int currCol, vector<vector<bool>>& ocean) {
        ocean[currRow][currCol] = 1;

        vector<vector<int>> direc = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};

        for (int dir = 0; dir < 4; dir++) {
            int newRow = currRow + direc[dir][0];
            int newCol = currCol + direc[dir][1];

            if (isSafe(newRow, newCol, row, col) and
                ocean[newRow][newCol] == 0 and
                heights[currRow][currCol] <= heights[newRow][newCol]) {
                dfs(heights, row, col, newRow, newCol, ocean);
            }
        }
    }

    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        vector<vector<int>> res;

        int row = heights.size();
        int col = heights[0].size();

        vector<vector<bool>> pac(row, vector<bool>(col, 0));
        vector<vector<bool>> atl(row, vector<bool>(col, 0));

        for (int i = 0; i < row; i++) {
            dfs(heights, row, col, i, 0, pac);
            dfs(heights, row, col, i, col - 1, atl);
        }

        for (int j = 0; j < col; j++) {
            dfs(heights, row, col, 0, j, pac);
            dfs(heights, row, col, row - 1, j, atl);
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (pac[i][j] and atl[i][j]) {
                    res.push_back({i, j});
                }
            }
        }

        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Rotting Oranges (https://leetcode.com/problems/rotting-oranges/)

Statement: Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

```cpp

class Solution {
public:
    bool isSafe(int newRow, int newCol, int row, int col) {
        int checkRow = (newRow >= 0 and newRow < row);
        int checkCol = (newCol >= 0 and newCol < col);

        return checkRow and checkCol;
    }

    int orangesRotting(vector<vector<int>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        int freshCount = 0, timeCount = 0;

        queue<pair<pair<int, int>, int>> q;

        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                if(grid[i][j] == 2) {
                    q.push({{i, j}, 0});
                } else if(grid[i][j] == 1) {
                    freshCount++;
                }
            }
        }

        if(freshCount == 0) return 0;

        while(!q.empty()) {
            int levelSize = q.size();

            for(int sz = 0; sz < levelSize; sz++) {
                int currRow = q.front().first.first;
                int currCol = q.front().first.second;
                int currTimeCount = q.front().second;
                q.pop();

                vector<vector<int>> direc = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
                for(int dir = 0; dir < 4; dir++) {
                    int newRow = currRow + direc[dir][0];
                    int newCol = currCol + direc[dir][1];

                    if(isSafe(newRow, newCol, row, col) and grid[newRow][newCol] == 1) {
                       q.push({{newRow, newCol,}, currTimeCount + 1});
                       grid[newRow][newCol] = 2;
                       freshCount--;
                       timeCount = max(timeCount, currTimeCount + 1);
                    }
                }
            }
        }

        return freshCount > 0 ? -1 : timeCount;
    }
};

```
------------------------------------------------------------------------------------------------------------------------

01 Matrix (https://leetcode.com/problems/01-matrix/description/)

Statement: Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
The distance between two cells sharing a common edge is 1.

Input: mat = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[0,0,0],[0,1,0],[0,0,0]]

```cpp

class Solution {
public:
    void dfs(int i, int j, vector<vector<int>>& mat, vector<vector<int>>& dist) {
        vector<vector<int>> direc = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};

        for (int dir = 0; dir < 4; dir++) {
            int ni = i + direc[dir][0];
            int nj = j + direc[dir][1];

            if (ni >= 0 && ni < mat.size() && nj >= 0 && nj < mat[0].size()) {
                if (dist[ni][nj] > dist[i][j] + 1) {
                    dist[ni][nj] = dist[i][j] + 1;
                    dfs(ni, nj, mat, dist);
                }
            }
        }
    }

    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int m = mat.size();
        int n = mat[0].size();

        vector<vector<int>> dist(m, vector<int>(n, INT_MAX));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) {
                    dist[i][j] = 0;
                    dfs(i, j, mat, dist);
                }
            }
        }

        return dist;
    }
};

//Smart Solution with O(1) space
 class Solution {
 public:
     vector<vector<int>> updateMatrix(vector<vector<int>> &mat) {
         int m = mat.size(), n = mat[0].size(), INF = m + n; // The distance of cells is up to (M+N)
         for (int r = 0; r < m; r++) {
             for (int c = 0; c < n; c++) {
                 if (mat[r][c] == 0) continue;
                 int top = INF, left = INF;
                 if (r - 1 >= 0) top = mat[r - 1][c];
                 if (c - 1 >= 0) left = mat[r][c - 1];
                 mat[r][c] = min(top, left) + 1;
             }
         }
         for (int r = m - 1; r >= 0; r--) {
             for (int c = n - 1; c >= 0; c--) {
                 if (mat[r][c] == 0) continue;
                 int bottom = INF, right = INF;
                 if (r + 1 < m) bottom = mat[r + 1][c];
                 if (c + 1 < n) right = mat[r][c + 1];
                 mat[r][c] = min(mat[r][c], min(bottom, right) + 1);
             }
         }
         return mat;
     }
 };

//https://leetcode.com/problems/01-matrix/solutions/3617748/why-2-pass-dp-works-using-pictorial-explanation/

```
------------------------------------------------------------------------------------------------------------------------

 Subsets (https://leetcode.com/problems/subsets/description/)

 Input: nums = [1,2,3]
 Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

 Input: nums = [0]
 Output: [[],[0]]


```cpp

 class Solution {
 public:
     void solve(vector<int>& nums, vector<vector<int>>& res, vector<int> curr, int idx) {
         if (idx == nums.size()) {
             res.push_back(curr);
             return;
         }

         // Include the current element (add it to curr and move to next)
         curr.push_back(nums[idx]);
         solve(nums, res, curr, idx + 1);

         // Skip duplicates
         while (idx + 1 < nums.size() && nums[idx] == nums[idx + 1]) {
             idx++;
         }

         // Exclude the current element (do not add it to curr, just move to
         // next)
         curr.pop_back();
         solve(nums, res, curr, idx + 1);
     }

     vector<vector<int>> subsets(vector<int>& nums) {

         vector<vector<int>> res;
         vector<int> curr;

         solve(nums, res, curr, 0);

         return res;
     }
 };
```
------------------------------------------------------------------------------------------------------------------------

Subsets II (https://leetcode.com/problems/subsets-ii/)

Given an integer array nums that may contain duplicates, return all possible
subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.

```cpp

class Solution {
public:
    void solve(vector<int>& nums, vector<vector<int>>& res, vector<int> curr, int idx) {
        if (idx == nums.size()) {
            res.push_back(curr);
            return;
        }

        // Include the current element (add it to curr and move to next)
        curr.push_back(nums[idx]);
        solve(nums, res, curr, idx + 1);
        curr.pop_back();
        // Skip duplicates
        while (idx + 1 < nums.size() && nums[idx] == nums[idx + 1]) {
            idx++;
        }

        // Exclude the current element (do not add it to curr, just move to
        // next)

        solve(nums, res, curr, idx + 1);
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> curr;
        sort(begin(nums), end(nums));

        solve(nums, res, curr, 0);

        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Permutation with Spaces (https://www.geeksforgeeks.org/problems/permutation-with-spaces3627/1)

Input:
s = "ABC"
Output: (A B C)(A BC)(AB C)(ABC)
Explanation:
ABC
AB C
A BC
A B C
These are the possible combination of "ABC".

```cpp

void solve(string& s, int n, int idx, string& curr, vector<string>& res) {
    if (idx == n) {
        res.push_back(curr);
        return;
    }

    if (idx > 0) {
        curr.push_back(' ');
        curr.push_back(s[idx]);
        solve(s, n, idx + 1, curr, res);
        curr.pop_back();
        curr.pop_back();
    }

    curr.push_back(s[idx]);
    solve(s, n, idx + 1, curr, res);
    curr.pop_back();
}

vector<string> permutation(string s) {
    int n = s.size();
    vector<string> res;
    string curr;

    curr.push_back(s[0]);

    solve(s, n, 1, curr, res);

    sort(res.begin(), res.end());
    return res;
}
```
------------------------------------------------------------------------------------------------------------------------

Letter Case Permutation (https://leetcode.com/problems/letter-case-permutation/)

Statement:Given a string s, you can transform every letter individually to be lowercase or uppercase to create another string.
Return a list of all possible strings we could create. Return the output in any order.
Input: s = "a1b2"
Output: ["a1b2","a1B2","A1b2","A1B2"]

```cpp
class Solution {
public:
    void solve(string& s, int len, int idx, string& curr, vector<string>& res) {
        if (idx == len) {
            res.push_back(curr);
            return;
        }

        if ((s[idx] >= 'a' and s[idx] <= 'z') or
            (s[idx] >= 'A' and s[idx] <= 'Z')) {

            if (s[idx] >= 'A' and s[idx] <= 'Z')
                s[idx] += 32;
            curr.push_back(s[idx]);
            solve(s, len, idx + 1, curr, res);
            curr.pop_back();

            if (s[idx] >= 'a' and s[idx] <= 'z')
                s[idx] -= 32;
            curr.push_back(s[idx]);
            solve(s, len, idx + 1, curr, res);
            curr.pop_back();

        } else {
            curr.push_back(s[idx]);
            solve(s, len, idx + 1, curr, res);
            curr.pop_back();
        }
    }

    vector<string> letterCasePermutation(string s) {
        int len = s.size();
        vector<string> res;
        string curr;
        solve(s, len, 0, curr, res);

        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Generate Parentheses (https://leetcode.com/problems/generate-parentheses/)
Statement: Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

```cpp

// without pass by reference (&) in curr
class Solution {
public:
    void dfs(int openP, int closedP, string curr, vector<string>& res, int n) {
        if (openP == closedP and openP + closedP == 2 * n) {
            res.push_back(curr);
            return;
        }

        if (openP < n) {
            dfs(openP + 1, closedP, curr + "(", res, n);
        }

        if (closedP < openP) {
            dfs(openP, closedP + 1, curr + ")", res, n);
        }
    }

    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string curr = "";
        dfs(0, 0, curr, res, n);

        return res;
    }
};

OR

    // with pass by reference (&) in curr, so need to backtrack too
    class Solution {
public:
    void dfs(int openP, int closedP, string& curr, vector<string>& res, int n) {
        if (openP == closedP and openP + closedP == 2 * n) {
            res.push_back(curr);
            return;
        }

        if (openP < n) {
            curr.push_back('(');
            dfs(openP + 1, closedP, curr, res, n);
            curr.pop_back();
        }

        if (closedP < openP) {
            curr.push_back(')');
            dfs(openP, closedP + 1, curr, res, n);
            curr.pop_back();
        }
    }

    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string curr = "";
        dfs(0, 0, curr, res, n);

        return res;
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Combination Sum (https : // leetcode.com/problems/combination-sum/)

```cpp

class Solution {
public:
    void solve(vector<int>& candidates, int idx, int n, int& target,
               vector<vector<int>>& res, vector<int>& curr, int sum) {
        if (sum == target) {
            res.push_back(curr);
            return;
        }

        if (target < sum || idx == n) {
            return;
        }

        sum += candidates[idx];
        curr.push_back(candidates[idx]);
        solve(candidates, idx, n, target, res, curr, sum);

        sum -= candidates[idx];
        curr.pop_back();

        solve(candidates, idx + 1, n, target, res, curr, sum);
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        int n = candidates.size();
        vector<vector<int>> res;
        vector<int> curr;
        solve(candidates, 0, n, target, res, curr, 0);

        return res;
    }
};

```
------------------------------------------------------------------------------------------------------------------------

Partitions with Given Difference (https://www.geeksforgeeks.org/problems/partitions-with-given-difference/1?itm_source=geeksforgeeks&itm_medium=article&itm_campaign=practice_card)

Statement: Given an array of integers and a difference,
The task is to find the number of ways to partition the array into two subsets such that the difference between the sum of the two subsets is equal to the given difference.
Input: arr[] =  [5, 2, 6, 4], d = 3
Output: 1
Explanation: There is only one possible partition of this array. Partition : {6, 4}, {5, 2}. The subset difference between subset sum is: (6 + 4) - (5 + 2) = 3.


```cpp

class Solution {
public:
    int solve(vector<int>& nums, int n, int target, vector<vector<int>>& dp) {
        for (int j = 0; j <= target; j++) {
            dp[0][j] = 0;
        }

        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= target; j++) {
                if (j >= nums[i - 1]) {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[n][target];
    }

    int countPartitions(vector<int>& arr, int diff) {
        int n = arr.size();
        int sum = 0, sum1 = 0;
        for (int i = 0; i < n; i++) {
            sum += arr[i];
        }

        if ((sum + diff) % 2 != 0) {
            return 0;
        }

        sum1 = ((diff + sum) >> 1);

        vector<vector<int>> dp(n + 1, vector<int>(sum1 + 1, -1));
        return solve(arr, n, sum1, dp);
    }
};
```
------------------------------------------------------------------------------------------------------------------------

Counting Bits (https://leetcode.com/problems/counting-bits/)

Statement: Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101


```cpp

class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> result(n + 1, 0);
        for (int i = 0; i <= n; i++) {
            if (!(i & 1)) {
                result[i] = result[i >> 1];
            } else {
                result[i] = result[i >> 1] + 1;
            }
        }

        return result;
    }
};

//Alternate solution using inbuild function:
 class Solution {
 public:
     vector<int> countBits(int n) {
         vector<int> result(n+1, 0);
         for(int i = 0; i <= n; i++) {
             result[i] = __builtin_popcount(i);
         }

         return result;
     }
 };
```
 ------------------------------------------------------------------------------------------------------------------------
