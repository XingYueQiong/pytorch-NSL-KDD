#下载说明
点击左上角的main，代码不在这个main分支，在xyq-pytorch-NSL-KDD分支，选择好分支后，点右上角下载

#最新更新
尝试了一下git branch，发现本地分支是默认main，我就直接git add把所有文件添加然后git push了，在这个main分支应该也可以下载使用了

小小总结一下git提交流程
git status查看修改
git add添加修改
git commit -m "test"

git branch查看分支
git branch -a查看所有分支
git push提价到默认的main分支
git push origin main:Xyq-Pytorch-NSL-KDD
git push -f origin main:Xyq-Pytorch-NSL-KDD强制修改


理论上来说是下面这句
git push <远程仓库名> <本地分支名>:<远程分支名>
git push pytorch-NSL-KDD main:Xyq-Pytorch-NSL-KDD
但是使用git remote -v查看发现，远程仓库名是origin，而pytorch-NSL-KDD只是路径
但可以修改名字：git remote rename <旧名称> <新名称>