1. Maximum Depth of Binary Tree (https://leetcode.com/problems/maximum-depth-of-binary-tree/?envType=study-plan-v2&envId=top-interview-150)


Solution: 

//DFS approach

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        
        int maxLeft = maxDepth(root->left);
        int maxRight = maxDepth(root->right);

        return max(maxLeft, maxRight) + 1;
    }
};

// BFS approach

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;

        queue<TreeNode*>que;
        que.push(root);

        int depth=0;

        while(!que.empty()){
            depth++;
            int n = que.size();
            while(n--){
                TreeNode* temp = que.front();
                que.pop();

                if(temp->left) que.push(temp->left);
                if(temp->right) que.push(temp->right);
            }
        }
        return depth;
    }
};

Bonus Question (ABOVE FOLLOW QUESTION)

Minimum Depth of Binary Tree(https://leetcode.com/problems/minimum-depth-of-binary-tree/)

Solution: 

//DFS approach

class Solution {
public:
    int minDepth(TreeNode* root) {
        if(!root) return 0;

        int L = minDepth(root->left);
        int R = minDepth(root->right);

        return (root->left and root->right) ? 1+min(L,R) : 1+max(L,R);
    }
};

//BFS approach

class Solution {
public:
    int minDepth(TreeNode* root) {
        if(!root) return 0;

        queue<TreeNode*>que;
        que.push(root);

        int depth=0;

        while(!que.empty()){
            depth++;
            int n = que.size();
            while(n--){
                TreeNode* temp = que.front();
                que.pop();

                if(!temp->left and !temp->right) return depth;

                if(temp->left) que.push(temp->left);
                if(temp->right) que.push(temp->right);
            }
        }
        return depth;
    }
};

2. Same Tree (https://leetcode.com/problems/same-tree/?envType=study-plan-v2&envId=top-interview-150)


Solution: 

class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(!p and !q) return true;

        if(!p || !q) return false;

        if(p->val != q->val) 
        return false;
        
        bool isSameLeft = isSameTree(p->left, q->left);
        bool isSameRight = isSameTree(p->right, q->right);

        return isSameLeft and isSameRight;
    }
};


3. Invert Binary Tree (https://leetcode.com/problems/invert-binary-tree/?envType=study-plan-v2&envId=top-interview-150)

Solution: 

//DFS approach

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root) return root;

        invertTree(root->left);
        invertTree(root->right);

        TreeNode* temp = root->left;
        root->left=root->right;
        root->right=temp;

        return root;
    }
};

// BFS approach

Solution: 

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root) return root;

        queue<TreeNode*>q;
        q.push(root);

        while(!q.empty()){
            int n = q.size();
            while(n--){
                TreeNode* temp = q.front();
                q.pop();

                TreeNode* node = temp->right;
                temp->right = temp->left;
                temp->left = node;

                if(temp->left) q.push(temp->left);
                if(temp->right) q.push(temp->right);
            }
        }
        return root;
    }
};


4.  Symmetric Tree(https://leetcode.com/problems/symmetric-tree/?envType=study-plan-v2&envId=top-interview-150)

Solution: 

class Solution {
public:

    bool isMirror(TreeNode* root1, TreeNode* root2){
        if(!root1 and !root2) return 1;
        if(root1 and root2 and root1->val == root2->val) return (isMirror(root1->left, root2->right) and isMirror(root1->right, root2->left));
    
    return 0;
    }

    bool isSymmetric(TreeNode* root) {
        return isMirror(root, root);
    }
};


5. Construct Binary Tree from Preorder and Inorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?envType=study-plan-v2&envId=top-interview-150)

Solution: 

class Solution {
public:

    TreeNode* solve(vector<int>&inorder, vector<int>&preorder, int start, int end, int &preIdx, map<int,int>&mp) {
        if(start > end) return NULL;

        TreeNode* root = new TreeNode(preorder[preIdx++]);
        int i = mp[root->val];

        root->left = solve(inorder, preorder, start, i-1, preIdx, mp);
        root->right = solve(inorder, preorder, i+1, end, preIdx, mp);
        
        return root;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = inorder.size();
        int preIdx = 0;
        map<int,int>mp;
        for(int i=0;i<inorder.size(); i++){
            mp[inorder[i]] = i;
        }
        return solve(inorder, preorder, 0, n-1, preIdx, mp);
    }
};


6. Construct Binary Tree from Inorder and Postorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/?envType=study-plan-v2&envId=top-interview-150)

(Ismai "root->right" pahale aya hai "root->left" se, solution mai because postorder mai root last mai aata hai aur root ke baad jo bhi aata node wo inorder mai right mai hai , so that's why first right then left recursive call)


Solution: 

class Solution {
public:

    TreeNode* solve(vector<int>&inorder, vector<int>&postorder, int start, int end, int &postIdx) {
        if(start > end) return NULL;

        TreeNode* root = new TreeNode(postorder[postIdx]);
        int i = start;
        while(inorder[i]!=postorder[postIdx]) i++;

        postIdx--;
        root->right = solve(inorder, postorder, i+1, end, postIdx);
        root->left = solve(inorder, postorder, start, i-1, postIdx);

        return root;
    }

    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        int n = inorder.size();
        int postIdx = n-1;

        return solve(inorder, postorder, 0, n-1, postIdx);
    }
};



7. Populating Next Right Pointers in Each Node II


Solution: 

class Solution {
public:
    Node* connect(Node* root) {
        if(!root) return root;
        queue<Node*>q;
        q.push(root);

        while(!q.empty()){
            int n = q.size();
            while(n--){
                Node* node = q.front();
                q.pop();
                if(n==0) node->next = NULL;
                else node->next = q.front();

                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
            }
        }
        return root;
    }
};


8. Flatten Binary Tree to Linked List (https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2&envId=top-interview-150)

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Solution: 

class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* node = root;
        while(node){
            if(node->left){
                TreeNode* rightMost = node->left;
                while(rightMost->right){
                    rightMost = rightMost->right;
                }
                rightMost->right = node->right;
                node->right = node->left;
                node->left = NULL;
            }
            node = node->right;
        }
    }
};


9.  Path Sum (https://leetcode.com/problems/path-sum/?envType=study-plan-v2&envId=top-interview-150)


Solution: 


class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(!root) return 0;

        if(root->val == targetSum and !root->left and !root->right) return 1;

        return hasPathSum(root->left, targetSum - root->val) || hasPathSum(root->right, targetSum - root->val);
    }
};


10. Sum Root to Leaf Numbers (https://leetcode.com/problems/sum-root-to-leaf-numbers/?envType=study-plan-v2&envId=top-interview-150)

Input: root = [4,9,0,5,1]
Output: 1026

Solution: 

class Solution {
public:

    int solve(TreeNode* root, int curr){
        if(!root) return 0;

        curr = (curr*10) + root->val;

        if(!root->left and !root->right) return curr;

        return solve(root->left, curr) + solve(root->right, curr);
    }

    int sumNumbers(TreeNode* root) {
        return solve(root, 0);
    }
};


11. Binary Tree Maximum Path Sum (https://leetcode.com/problems/binary-tree-maximum-path-sum/description/?envType=study-plan-v2&envId=top-interview-150)

Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.


Solution: 

class Solution {
public:
    int solve(TreeNode* root, int & res) {
        if (!root)
            return 0;

        int left_node = solve(root->left, res);
        int right_node = solve(root->right, res);

        int temp = max(max(left_node, right_node) + root->val, root->val);
        if(!root->left and !root->right) 
        temp = root->val; 
        
        int ans = max(temp, left_node + right_node + root->val);
        res = max(ans, res);
        return temp;
    }

    int maxPathSum(TreeNode* root) {
        int res = INT_MIN;
        solve(root, res);
        return res;
    }
};


12. Binary Search Tree Iterator(https://leetcode.com/problems/binary-search-tree-iterator/description/?envType=study-plan-v2&envId=top-interview-150)

Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]

Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False



Solution: 









13. Count Complete Tree Nodes (https://leetcode.com/problems/count-complete-tree-nodes/?envType=study-plan-v2&envId=top-interview-150)

(NOTE: Design an algorithm that runs in less than O(n) time complexity.)

Solution: 


class Solution {
public:
    int countNodes(TreeNode* root) {
        if(!root) return 0;

        int lh = findLeftHeight(root);
        int rh = findRightHeight(root);

        if(lh == rh) return (1<<lh)-1;

        return 1 + countNodes(root->left) + countNodes(root->right);
    }

    int findLeftHeight(TreeNode* root){
        if(!root) return 0;
        TreeNode* temp = root;

        int cnt = 0;
        while(temp){
            temp = temp->left;
            cnt++;
        }
        return cnt;
    }

    int findRightHeight(TreeNode* root){
        if(!root) return 0;
        TreeNode* temp = root;

        int cnt = 0;
        while(temp){
            temp = temp->right;
            cnt++;
        }
        return cnt;
    }
};


14. Lowest Common Ancestor of a Binary Tree (https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=top-interview-150)

Solution: 


class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;

        if(root == p || root == q) return root;

        TreeNode* ln = lowestCommonAncestor(root->left, p, q);
        TreeNode* rn =  lowestCommonAncestor(root->right, p, q);

        if(ln && rn) return root;

        if(ln) return ln;

        return rn;
    }
};


15. Binary Tree Right Side View (https://leetcode.com/problems/binary-tree-right-side-view/description/?envType=study-plan-v2&envId=top-interview-150)

Solution: 


class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> result;
        if (!root) return result;
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int size = q.size();
            int rightMostValue;
            for (int i = 0; i < size; ++i) {
                TreeNode* node = q.front();
                q.pop();
                if(i==size-1)
                rightMostValue = node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            result.push_back(rightMostValue);
        }
        
        return result;
    }
};



16. Average of Levels in Binary Tree (https://leetcode.com/problems/average-of-levels-in-binary-tree/?envType=study-plan-v2&envId=top-interview-150)

Solution: 


class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        queue<TreeNode*>q;
        q.push(root);

        vector<double>res;
        while(!q.empty()){
            int n = q.size();
            
            double sum = 0;
            int cnt = 0;
            while(n--){
                TreeNode* temp = q.front();
                q.pop();

                sum+= temp->val;
                cnt++;

                if(temp->left) q.push(temp->left);
                if(temp->right) q.push(temp->right);
            }
            res.emplace_back(sum/cnt);
        }
        return res;
    }
};


17. Binary Tree Level Order Traversal (https://leetcode.com/problems/binary-tree-level-order-traversal/?envType=study-plan-v2&envId=top-interview-150)

Solution: 


class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(!root) return {};
        queue<TreeNode*> q;
        q.push(root);
        vector<vector<int>>res;
        while(!q.empty()){
            int n = q.size();
            vector<int>v;
            while(n--){
                TreeNode* temp = q.front();
                q.pop();
                v.emplace_back(temp->val);

                if(temp->left) q.push(temp->left);
                if(temp->right) q.push(temp->right);
            }
            res.emplace_back(v);
        }
        return res;
    }
};

18.  Binary Tree Zigzag Level Order Traversal (https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal?envType=study-plan-v2&envId=top-interview-150)


Solution: 

class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root)
            return res;
        queue<TreeNode*> q;
        q.push(root);
        bool flag = 1;

        while (!q.empty()) {
            int n = q.size();
            vector<int> temp(n);
            for (int i = 0; i < n; i++) {
                TreeNode* node = q.front();
                q.pop();

                int idx = 0;
                if (flag)
                    idx = i;
                else
                    idx = (n - 1 - i);

                temp[idx] = node->val;

                if (node->left)
                    q.push(node->left);
                if (node->right)
                    q.push(node->right);
            }
            flag = !flag;
            res.emplace_back(temp);
        }
        return res;
    }
};


19. Maximum Level Sum of a Binary Tree

Input: root = [1,7,0,7,-8,null,null]
Output: 2
Explanation: 
Level 1 sum = 1.
Level 2 sum = 7 + 0 = 7.
Level 3 sum = 7 + -8 = -1.
So we return the level with the maximum sum which is level 2.


Solution: 


//DFS approach:

class Solution {
public:
    void solve(TreeNode* root, int level, int &max_level, vector<int>&v){
        if(!root) return;
        
        if (level >= v.size()) {
            v.resize(level + 1, 0);
        }

        v[level] += root->val;
        solve(root->left, level+1, max_level, v);
        solve(root->right, level+1, max_level, v);
        max_level = max(max_level, level);
    }

    int maxLevelSum(TreeNode* root) {
        vector<int>v;
        int max_level = 0, level = 0, sum = INT_MIN; 
        solve(root, level, max_level, v);

        for(int i=0; i<=max_level; i++){
            if(v[i] > sum){
                sum = v[i];
                level = i+1;
            }
        }
        return level;
    }
};



//BFS approach:

class Solution {
public:
    int solve(TreeNode* root, int& max_level, int& max_sum) {
        if (!root)
            return 0;

        queue<TreeNode*> q;
        q.push(root);

        int level = 0;
        while (!q.empty()) {
            int size = q.size();
            int sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode *node = q.front();
                q.pop(); 

                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);

                sum += node->val;
            }
            level++;
            if(max_sum < sum){
                max_sum = sum;
                max_level = level;
            }
        }
        return max_level;
    }

    int maxLevelSum(TreeNode* root) {
        int level = 0, max_sum = INT_MIN;
        return solve(root, level, max_sum);
    }
};







