# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:16:51 2022

@author: hren
"""

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import json
import os
import csv
import time
import math
#import nltk
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
import string
import datetime
import networkx as nx
import pandas as pd
import gensim
from gensim import corpora
from gensim import models
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score

github_url = 'https://api.github.com'
github_auth = {'Authorization': 'token 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'}
sess = requests.Session()

def get_repo(repos):
    file = open('../{}/github_data/all_repo_info.csv'.format('astropy'), 'w+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(["repo", "stars","watch","fork",'create_time','update_time'])
    
    for repo in repos:
        
        subjectUser=repo.split('/')[0]
        subjectProjectName=repo.split('/')[1]
        print(subjectUser,subjectProjectName)
        
        params = {"state": "all"}
        repo_url='/'.join([github_url,'repos',subjectUser,subjectProjectName])
       
        response = sess.get(repo_url, headers=github_auth)
        print(repo_url)
        repo_data=json.loads(response.text)
        print(repo_data['id'])
        full_name=repo_data['full_name']
        stars=repo_data['stargazers_count']
        watchs=repo_data['watchers_count']
        forks=repo_data['forks_count']
        create=repo_data['created_at']
        update=repo_data['updated_at']
        csv_writer.writerow([full_name,stars,watchs,forks,create,update])
        with open('../{}/github_data/repo_info.json'.format(subjectProjectName), "w") as f:
            json.dump(response.text,f,indent=4)
            f.close()

def get_comment(repos):
    
    for repo in repos:
        repo=repo.split('/')[-1]
        
        for file in os.listdir('../{}/github_data/issues/'.format(repo)):
            
            with open(os.path.join('../{}/github_data/issues/'.format(repo),file),'r',encoding='utf8') as file:
                issue=json.load(file)
                file.close()
            issue_number=issue['number']
            comment_url=issue['comments_url']
            
            for i in range(1,100):
                params = {'per_page':100,'page':i}
                response = sess.get(comment_url, params=params,headers=github_auth)
                
                comments=json.loads(response.text)
                for comment in tqdm(comments):
                    ids=comment['id']
                    if(os.path.exists('../{}/github_data/comments/{}_{}.json'.format(repo,str(issue_number),str(ids)))):
                        continue
                    with open('../{}/github_data/comments/{}_{}.json'.format(repo,str(issue_number),str(ids)), "w",encoding='utf8') as f:
                        json.dump(comment,f,indent=4)
                        f.close() 
        

def get_issues(repo_urls):
    for repo_url in repo_urls:
        subjectUser=repo_url.split('/')[0]
        subjectProjectName=repo_url.split('/')[1]
        print(subjectUser,subjectProjectName)
        
        repo_url='/'.join([github_url,'repos',subjectUser,subjectProjectName])
        response = sess.get(repo_url, headers=github_auth)
        print(repo_url)
        repo_data=json.loads(response.text)
        create=repo_data['created_at']
        issues_url='/'.join([github_url,'repos',subjectUser,subjectProjectName,'issues'])
        
        for i in range(1,10001):
            params = {"state": "closed","since":create,'per_page':100,'page':i}
            response = sess.get(issues_url, params=params,headers=github_auth)
            issues=json.loads(response.text)
            for issue in tqdm(issues):
                number=issue['number']
                if(os.path.exists('../{}/github_data/issues/{}_info.json'.format(subjectProjectName,str(number)))):
                    continue
                with open('../{}/github_data/issues/{}_info.json'.format(subjectProjectName,str(number)), "w") as f:
                    json.dump(issue,f,indent=4)
                    f.close() 
               
def get_commit(repos):
    
    count=0
    k=0
    errors=0
    for repo in repos:
        subjectUser=repo.split('/')[0]
        subjectProjectName=repo.split('/')[1]
        repo_url='/'.join([github_url,'repos',subjectUser,subjectProjectName])  
        response = sess.get(repo_url, headers=github_auth)
        print(repo_url)
        repo_data=json.loads(response.text)
        create=repo_data['created_at']
        
        issues_url='/'.join([github_url,'repos',subjectUser,subjectProjectName,'commits'])
        
        for i in range(1,10001):
            params = {"state": "closed","since":create,'per_page':100,'page':i}
            print(params)
        
            response = sess.get(issues_url, params=params,headers=github_auth)
            print(response.url)
            
            commits=json.loads(response.text)
            if(len(commits)==0 and response.status_code==200):
                print('{} finish...........'.format(str(i)))
                break
            for commit in tqdm(commits):
                number=commit['sha']
                if(os.path.exists('../{}/github_data/commits/{}_info.json'.format(subjectProjectName,str(number)))):
                    continue
                with open('../{}/github_data/commits/{}_info.json'.format(subjectProjectName,str(number)), "w") as f:
                    json.dump(commit,f,indent=4)
                    f.close()
        
def get_repo(repos):
    file = open('../{}/github_data/all_repo_info.csv'.format('astropy'), 'w+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    #if(not exist_querys):
    csv_writer.writerow(["repo", "stars","watch","fork",'create_time','update_time'])
    
    for repo in repos:
        
        subjectUser=repo.split('/')[0]
        subjectProjectName=repo.split('/')[1]
        print(subjectUser,subjectProjectName)
        
        params = {"state": "all"}
        repo_url='/'.join([github_url,'repos',subjectUser,subjectProjectName])
       
        response = sess.get(repo_url, headers=github_auth)
        print(repo_url)
        repo_data=json.loads(response.text)
        print(repo_data['id'])
        full_name=repo_data['full_name']
        stars=repo_data['stargazers_count']
        watchs=repo_data['watchers_count']
        forks=repo_data['forks_count']
        create=repo_data['created_at']
        update=repo_data['updated_at']
        csv_writer.writerow([full_name,stars,watchs,forks,create,update])
        with open('../{}/github_data/repo_info.json'.format(subjectProjectName), "w") as f:
            json.dump(response.text,f,indent=4)
            f.close()

def get_related(content,repo):
    '''
    possible format:
        #6776  $$this lead to some false positive
        njues/project1#1
        https://github.com/numpy/numpy/issues/1683    https://github.com/numpy/numpy/issues/5241#issuecomment-125405573
        https://github.com/scipy/scipy/pull/58
        gh-1683
    return issues:
        [repo_name#6776, njues/project1#1, numpy/numpy#1683, repo_name#1683]
    '''
    print("get related issues......")
    #repo_name = user + '/' + projectName
    issues = []
    formats = ['\W(#[1-9]+[0-9]*)',
               '(\w+\/\w+#[1-9]+[0-9]*)',
               'github.com\/(\w+\/\w+)\/(issues|pull)\/([1-9]+[0-9]*)',
               'gh-([1-9]+[0-9]*)']
    for i in range(len(formats)):
        pattern = re.compile(formats[i])
        match = pattern.findall(content)
        for m in match:
            if i==0:
                issues += ([repo + m])
            elif i==1:
                issues += ([m])
            elif i==2:
                issues += ([m[0] + '#' + m[2]])
            elif i==3:
                issues += ([repo + '#' + m])

        issues += ([m[0] + '#' + m[2]])
    
    cross_issues = []
    within_issues=[]
    for issue in issues:
        if repo not in issue:
            cross_issues.append(issue)
        else:
            within_issues.append(issue)
    
    return (cross_issues,within_issues)


def get_cpc_issues(repos):
    for repo in repos:
        print(repo)
        repo_name=repo.split('/')[1]
        user=repo.split('/')[0]
        
        file = open('../{}/github_data/relate_issues.csv'.format(repo_name), 'w+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow(["issue",'related_issue','type','time','developer'])
        
        for file in os.listdir('../{}/github_data/comments/'.format(repo_name)):
            if(file.split('.')[-1]!='json'):
                continue
            issue=file.split('_')[0]
            comments=json.load(open(os.path.join('../{}/github_data/comments'.format(repo_name),file),encoding='utf8'))
            #print(comments)
            developer=comments['user']['login']
            create_time=comments['created_at']
            body=comments['body']
            print('{}......{}......comment......{}......{}......'.format(repo_name,issue,file.split('_')[1],body))
            cross_issues,within_issues=get_related(body, repo)
            print(cross_issues,within_issues)
            
            for r_issue in cross_issues:
                csv_writer.writerow([repo+'#'+issue,r_issue,'cross',create_time,developer])
            for r_issue in within_issues:
                csv_writer.writerow([repo+'#'+issue,r_issue,'within',create_time,developer])
            #break
        for file in os.listdir('../{}/github_data/issues/'.format(repo_name)):
            if(file.split('.')[-1]!='json'):
                continue
            issue=file.split('_')[0]
            comments=json.load(open(os.path.join('../{}/github_data/issues'.format(repo_name),file),encoding='utf8'))
            
            developer=comments['user']['login']
            create_time=comments['created_at']
            body=comments['body']
            title=comments['title']  
            
            print('{}......{}......title......{}......'.format(repo_name,issue,body))
            cross_issues,within_issues=get_related(title, repo)
            print(cross_issues,within_issues)
            for r_issue in cross_issues:
                csv_writer.writerow([repo+'#'+issue,r_issue,'cross',create_time,developer])
            for r_issue in within_issues:
                csv_writer.writerow([repo+'#'+issue,r_issue,'within',create_time,developer])
            
            print('{}......{}......body......{}......'.format(repo_name,issue,body))
            if(body):
                cross_issues,within_issues=get_related(body, repo)
                print(cross_issues,within_issues)
                for r_issue in cross_issues:
                    csv_writer.writerow([repo+'#'+issue,r_issue,'cross',create_time,developer])
                for r_issue in within_issues:
                    csv_writer.writerow([repo+'#'+issue,r_issue,'within',create_time,developer])

def get_new_cpc_issues(repos):
    
    for repo in repos:
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        file=open('../{}/github_data/add_indirect_issue_labels.csv'.format(project),'w',newline='',encoding='utf8')
        header=['issue','label']
        file_csv=csv.writer(file)
        file_csv.writerow(header)
        
        edges=[]
        with open('../{}/github_data/relate_issues.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                if(row['type']=='within'):
                    edges.append((row['issue'],row['related_issue']))
            file.close()
        issue_labels={}
        nodes=[]
        cpc_issues=[]
        non_cpc_issues=[]
        with open('../{}/github_data/total_issues.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                nodes.append(row['issue'])
                issue_labels[row['issue']]=row['label']
                if(row['label']=='1'):
                    cpc_issues.append(row['issue'])
                else:
                    non_cpc_issues.append(row['issue'])
            file.close()        
        
        g=nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        #print(nx.info(g))
        new_issue_label={}
        for non_issue in non_cpc_issues:
            label=0
            for issue in cpc_issues:
                if(nx.has_path(g,non_issue,issue)):
                    label=1
                    break
            #print(repo,non_issue,label)
            new_issue_label[non_issue]=label
        
        for issue,label in issue_labels.items():
            if(label=='1'):
                file_csv.writerow([issue,label])
                continue
            file_csv.writerow([issue,new_issue_label[issue]])
    

def get_issue_experience(repos):
    
    for repo in repos:
        
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        with open('../{}/github_data/all_repo_issues.json'.format(repo),'r',encoding='utf8') as file:
            total_issues=json.load(file)
            file.close()
        
        with open('../{}/github_data/all_repo_issue_indirect_labels.json'.format(repo),'r',encoding='utf8') as file:
            total_issue_labels=json.load(file)
            file.close()
        
        with open('../{}/github_data/all_repo_issue_times.json'.format(repo),'r',encoding='utf8') as file:
            total_issue_time=json.load(file)
            file.close()
        
        with open('../{}/github_data/all_repo_developer_issues.json'.format(repo),'r',encoding='utf8') as file:
            total_developer_issues=json.load(file)
            file.close()
        
        with open('../{}/github_data/all_repo_issue_developers.json'.format(repo),'r',encoding='utf8') as file:
            total_issue_developer=json.load(file)
            file.close()
        
        with open('../{}/github_data/all_repo_developer_p_issues.json'.format(repo),'r',encoding='utf8') as file:
            total_developer_p_issues=json.load(file)
            file.close()
        with open('../{}/github_data/all_repo_developer_comment_times.json'.format(repo),'r',encoding='utf8') as file:
            total_developer_comment_time=json.load(file)
            file.close()
        
        file=open('../{}/github_data/issue_developer_experience.csv'.format(project),'w',newline='',encoding='utf8')
        header=['issue','total_report_issue_num','total_report_cpc_issue_num','total_participant_issue_num','total_participant_cpc_issue_num']
        file_csv=csv.writer(file)
        file_csv.writerow(header)
        
        new_total_issue_time={}
        for issue,time in total_issue_time.items():
            new_total_issue_time[issue]=datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        total_issue_times=sorted(new_total_issue_time.items(), key = lambda kv:(kv[1], kv[0]))
        total_sorted_issues=[]
        for issue,time in total_issue_times:
            total_sorted_issues.append(issue)
        
        with open('../{}/github_data/total_issues.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                developer=total_issue_developer[row['issue']]
                report_issues=total_developer_issues[developer]
                
                index=total_sorted_issues.index(row['issue'])
                prior_issues=total_sorted_issues[:index+1]
                report_num=len(set(report_issues) & set(prior_issues))
                
                cpc_issues=[]
                for issue in report_issues:
                    if(total_issue_labels[issue]=='1'):
                        cpc_issues.append(issue)
                report_cpc_num=len(set(cpc_issues)&set(prior_issues))
                
                time=new_total_issue_time[row['issue']]
                all_p_issues=total_developer_p_issues[developer]
                p_issues=[]
                cpc_p_issues=[]
                for p_issue in all_p_issues:
                    if(p_issue in report_issues):
                        p_time=new_total_issue_time[p_issue]
                        diff=time-p_time
                        if(diff.days>=0):
                            p_issues.append(p_issue)
                            if(total_issue_labels[p_issue]=='1'):
                                cpc_p_issues.append(p_issue)
                        continue
                    for comment_time in total_developer_comment_time[developer+'_'+p_issue]:
                        diff=time-datetime.datetime.strptime(comment_time, '%Y-%m-%d %H:%M:%S')
                        if(diff.days>=0):
                            p_issues.append(p_issue)
                            if(total_issue_labels[p_issue]=='1'):
                                cpc_p_issues.append(p_issue)
                            break
                #print(row['issue'],report_num,report_cpc_num,len(set(p_issues)),len(set(cpc_p_issues)))
                file_csv.writerow([row['issue'],report_num,report_cpc_num,len(set(p_issues)),len(set(cpc_p_issues))])

def get_clean(words):
    
    tokens=['\\r\\n','\\n','\\','=',"'",'`','//','/','*','\\\\',"\\'",'\\\\\\\\','`',"'",'0'
            ,'**','.','__','_','-',',','=+','_','^','^^','+',"'","\.",'~~','||','|','~',
            '1','2','3','4','5','6','7','8','9','FALSE','TRUE']
    new_temp=[]
    stops=stopwords.words('english')
    for word in words:
        if(word):
            word=word.lower()
            if word in stops:
                continue
            if word in string.punctuation:
                continue
            if(re.match(r'[+-]?\d+\.?\d*',word)):
                continue
            if(re.match(r'=[0-9]+',word)):
                continue
            if(re.match(r'[0-9]+',word)):
                continue
            if(re.match(r'[0-9]+\.[0-9]+',word)):
                continue
            if(re.match(r'[0-9]+\.[0-9]+E\+[0-9]+',word)):
                continue
            if(re.match(r'\d+\.\d+',word)):
                continue
            if(re.match(r'-\d+\.\d+',word)):
                continue
            if(len(word)==1):
                continue
            if('/' in word):
                new_temp.extend(word.split('/'))
                continue
            if(':' in word):
                new_temp.extend(word.split(':'))
                continue
            if('_' in word):
                new_temp.extend(word.split('_'))
                continue
            if('.' in word):
                new_temp.extend(word.split('.'))
                continue
            if('-' in word):
                new_temp.extend(word.split('-'))
                continue
            if('*' in word):
                new_temp.extend(word.split('*'))
                continue
            
            flag=False
            for token in tokens:
                if(token in word):
                    words=word.split(token)
                    flag=True
                    break
            if(flag):
                new_temp.extend(words)
            else:
                new_temp.append(word)
    return new_temp


def get_token(text):
    
    pattern = r"""(?x)                   # set flag to allow verbose regexps 
	            (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
	            |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
	            |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe 
	            |\.\.\.                # ellipsis 
	            #|(?:[.,;"'?():-_`])    # special characters with meanings 
	            """ 
    
    if(not text):
        return []
    new_temp=nltk.regexp_tokenize(text.lower().replace('\\r\\n',' '), pattern)
    '''
    new_temp=[]
    sens=nltk.sent_tokenize(text)
    for sen in sens:
        new_temp.extend(nltk.word_tokenize(sen))
    '''   
    #stops=stopwords.words('english')
    for i in range(12):
        new_temp=get_clean(new_temp)
    
    porter_stemmer=PorterStemmer()
    new=[]
    for word in new_temp:
        if(not word):
            continue
        if(len(word)<=1):
            continue
        #if(not re.match(r'[0-9]+',word)):
        new.append(porter_stemmer.stem(word))
    #print(new)   
    return(new)
                

def get_issue_text_metric(repos):
    
    formats = ['\W(#[1-9]+[0-9]*)',
               '([A-Za-z0-9_-]+\/[A-Za-z0-9_-]+#[1-9]+[0-9]*)',
               'github.com\/([A-Za-z0-9_-]+\/[A-Za-z0-9_-]+)\/(issues|pull)\/([1-9]+[0-9]*)',
               'gh-([1-9]+[0-9]*)']
    
    developer_format='@([A-Za-z0-9_-]+)'
    
    all_repos=[]
    with open('../repo.csv','r',encoding='utf8') as file:
        reader=csv.DictReader(file)
        for row in reader:
            #all_repos.append(row['repo'])
            all_repos.append(row['repo'].split('/')[1])
        file.close()
    #all_repos.extend(repos)
    for repo in repos:
        all_repos.append(repo.split('/')[1])
    
    for repo in repos:
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        file=open('../{}/github_data/new_issue_text_metric.csv'.format(project),'w',newline='',encoding='utf8')
        header=['issue','title_length','body_length','project_num','issue_num','devloper_num']
        file_csv=csv.writer(file)
        file_csv.writerow(header)
        
        with open('../{}/github_data/total_issues.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                issue_number=row['issue'].split('#')[1]
                issues=json.load(open('../{}/github_data/issues/{}_info.json'.format(project,issue_number),'r',encoding='utf8'))
                
                title=issues['title']
                orig_title_words=[word for word in re.split(r'(?:[;+,.#@!\\&\?\<\>`=\*\-\_\(\)/:\[\]\s\'\']\s*)',title.lower()) if word]
                title_words=get_token(title.lower())
                body=issues['body']
                if(not body):
                    body=''
                body_words=get_token(body.lower())
                orig_body_words=[word for word in re.split(r'(?:[;+,.#@!\\&\?\<\>`=\*\-\_\(\)/:\[\]\s\'\']\s*)',body.lower()) if word]
                developer=issues['user']['login']
                time=issues['created_at']
                print(project,issue_number)
                
                exist_repos=[]
                exist_repo_nums=[]
                for all_repo in set(all_repos):
                    temp=0
                    all_repo=all_repo.lower()
                    if(all_repo in orig_title_words):
                        temp+=orig_title_words.count(all_repo)
                    if(all_repo in orig_body_words):
                        temp+=orig_body_words.count(all_repo)
                    if(temp):
                        exist_repos.append(all_repo)
                        exist_repo_nums.append(temp)
                
                exist_issues=[]
                for i in range(len(formats)):
                    pattern = re.compile(formats[i])
                    match = pattern.findall(body)
                    for m in match:
                        if i==0:
                            temp=repo+m
                        elif i==1:
                            temp=m
                        elif i==2:
                            temp=m[0] + '#' + m[2]
                        elif i==3:
                            temp=repo + '#' + m
                        exist_issues.append(temp)
                    match = pattern.findall(title)
                    for m in match:
                        if i==0:
                            temp=repo+m
                        elif i==1:
                            temp=m
                        elif i==2:
                            temp=m[0] + '#' + m[2]
                        elif i==3:
                            temp=repo + '#' + m
                        exist_issues.append(temp)
                #print(exist_issues)
                
                exist_developers=[]
                pattern=re.compile(developer_format)
                match=pattern.findall(body)
                for m in match:
                    exist_developers.append(m)
                match=pattern.findall(title)
                for m in match:
                    exist_developers.append(m)
                #print(exist_developers)
                
                file_csv.writerow([row['issue'],len(title_words),len(body_words),len(exist_repos),len(set(exist_issues)),len(set(exist_developers))])
                
def get_issue_famility(repos):
    
    for repo in repos:
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        file=open('../{}/github_data/issue_famility.csv'.format(project),'w',newline='',encoding='utf8')
        header=['issue','issue_ratio','commit_ratio']
        file_csv=csv.writer(file)
        file_csv.writerow(header)
        
        #issue_participants={}
        developer_p_issues={}
        developer_comment_time={}
        for file in os.listdir('../{}/github_data/comments/'.format(project)):
            if(file.split('.')[-1]!='json'):
                continue
            issue=repo+'#'+file.split('_')[0]
            comments=json.load(open(os.path.join('../{}/github_data/comments'.format(project),file),'r',encoding='utf8'))
            developer=comments['user']['login']
            if(developer in developer_p_issues.keys()):
                developer_p_issues[developer].append(issue)
            else:
                developer_p_issues[developer]=[issue]
            time=comments['created_at'].replace('T',' ').replace('Z','')
            time=datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            key=developer+'_'+issue
            if(key in developer_comment_time.keys()):
                developer_comment_time[key].append(time)
            else:
                developer_comment_time[key]=[time]
        #print(developer_comment_time)
        issue_time={}
        developer_issues={}
        total_issues=[]
        issue_developer={}
        with open('../{}/github_data/total_issues.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                total_issues.append(row['issue'])
                issue_number=row['issue'].split('#')[1]
                issues=json.load(open('../{}/github_data/issues/{}_info.json'.format(project,issue_number),'r',encoding='utf8'))
                time=issues['created_at'].replace('T',' ').replace('Z','')
                time=datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                issue_time[row['issue']]=time
                developer=issues['user']['login']
                issue_developer[row['issue']]=developer
                if(developer in developer_issues.keys()):
                    developer_issues[developer].append(row['issue'])
                else:
                    developer_issues[developer]=[row['issue']]
                if(developer in developer_p_issues.keys()):
                    developer_p_issues[developer].append(row['issue'])
                else:
                    developer_p_issues[developer]=[row['issue']]
                
            file.close()
        issue_times=sorted(issue_time.items(), key = lambda kv:(kv[1], kv[0]))
        sorted_issues=[]
        for issue,time in issue_times:
            sorted_issues.append(issue)
        #print(sorted_issues)
        #break
        developer_commits={}
        commit_times={}
        for file in os.listdir('../{}/github_data/commits/'.format(project)):
            commits=json.load(open(os.path.join('../{}/github_data/commits'.format(project), file),'r',encoding='utf8'))
            commit=commits['sha']
            print(commit)
            time=commits['commit']['author']['date'].replace('T',' ').replace('Z','')
            time=datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            commit_times[commit]=time
            if(commits['author']):
                developer=commits['author']['login']
            else:
                developer=commits['commit']['author']['name']
            if(developer in developer_commits.keys()):
                developer_commits[developer].append(commit)
            else:
                developer_commits[developer]=[commit]
        commit_times=sorted(commit_times.items(), key = lambda kv:(kv[1], kv[0])) 
        
        issue_nums={}
        issue_ratios={}
        issue_participant_num={}
        issue_participant_ratio={}
        commit_nums={}
        commit_ratios={}
        for issue in total_issues:
            time=issue_time[issue]
            key=issue+'_'+time.strftime('%Y-%m-%d %H:%M:%S')
            if(key in issue_nums.keys()):
                file_csv.writerow([issue,issue_nums[key],issue_ratios[key],issue_participant_num[key],issue_participant_ratio[key],commit_nums[key],commit_ratios[key]])
                continue
            
            developer=issue_developer[issue]
            report_issues=developer_issues[developer]
            index=sorted_issues.index(issue)
            all_issues=sorted_issues[:index+1]
            issue_ratio=len(set(report_issues) & set(all_issues))/len(set(all_issues))
            
            p_issues=developer_p_issues[developer]
            exist_p_issues=[]
            #print(issue,developer)
            #print(set(p_issues))
            for p_issue in set(p_issues)&set(all_issues):
                if(p_issue in report_issues):
                    comment_time=issue_time[p_issue]
                    diff=time-comment_time
                    if(diff.days>=0):
                        exist_p_issues.append(p_issue)
                    #exist_p_issues.append(p_issue)
                    continue
                #flag=False
                for comment_time in developer_comment_time[developer+'_'+p_issue]:
                    diff=time-comment_time
                    if(diff.days>=0):
                        exist_p_issues.append(p_issue)
                        break
            participant_num=len(set(exist_p_issues))
            participant_ratio=len(set(exist_p_issues))/len(set(all_issues))
                 
            report_commits=[]
            if(developer in developer_commits.keys()):
                report_commits=developer_commits[developer]
            all_commits=[]
            if(len(report_commits)!=0):
                for commit,c_time in commit_times:
                    diff=time-c_time
                    if(diff.days>=0):
                        all_commits.append(commit)
                    else:
                        break
                if(len(all_commits)==0):
                    commit_ratio=0
                    #ommit_num=0
                else:
                    commit_ratio=len(set(report_commits)&set(all_commits))/len(set(all_commits))
            else:
                commit_ratio=0
                #commit_num=0
            #print(repo,issue,len(set(report_issues) & set(all_issues)),issue_ratio,participant_num,participant_ratio,len(set(report_commits)&set(all_commits)),commit_ratio)
            file_csv.writerow([issue,participant_ratio,commit_ratio])
            
            issue_participant_ratio[key]=participant_ratio
            commit_ratios[key]=commit_ratio

def train_process_model(repos):
    for repo in repos:
        project=repo.split('/')[1]
        issue_datas={}
        with open('../{}/github_data/issue_process_metrics.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.reader(file)
            for row in reader:
                issue_datas[row[0]]=row[1:]
            file.close()
        
        
        for i in range(5,10):
            train_datas=[]
            train_labels=[]
            #print(i,len(train_datas),len(train_datas[0]),len(train_labels))
            for j in range(i):
                file_path='../{}/github_data/word_embedding/fold_{}_dimension_{}_issue_labels.csv'.format(project,str(j),str(50))
                file_data=pd.read_csv(file_path)
                issue_label=dict(zip(file_data['issue'],file_data['label']))
                for issue,label in issue_label.items():
                    train_labels.append(int(label))
                    train_datas.append(issue_datas[issue])
            #min_max_scaler = preprocessing.MinMaxScaler()
            #train_datas=min_max_scaler.fit_transform(train_datas)
            
            file_path='../{}/github_data/word_embedding/fold_{}_dimension_{}_issue_labels.csv'.format(project,str(i),str(50))
            file_data=pd.read_csv(file_path)
            issue_label=dict(zip(file_data['issue'],file_data['label']))
            test_datas=[]
            test_labels=[]
            test_issues=[]
            for issue,label in issue_label.items():
                test_labels.append(int(label))
                test_issues.append(issue)
                test_datas.append(issue_datas[issue])
        
            
            for t in range(10):
                
                rf_clf=RandomForestClassifier(n_estimators=100)
                lr_clf=LogisticRegression()
                gb_clf=GradientBoostingClassifier()
                
                rf_clf.fit(train_datas,train_labels)
                lr_clf.fit(train_datas,train_labels)
                gb_clf.fit(train_datas,train_labels)
                
                
                scores=rf_clf.predict_proba(test_datas)
                file=open('../{}/github_data/result/{}/time_{}_fold_{}_rondom_forest_result.csv'.format(project,'process',str(t),str(i)),'w',newline='',encoding='utf8')
                header=['issue','0_prob','1_prob','label']
                file_csv=csv.DictWriter(file,header)
                file_csv.writeheader()
                
                for k in range(len(scores)):
                    #print('rf',issues[i])
                    file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})
                scores=lr_clf.predict_proba(test_datas)
                file=open('../{}/github_data/result/{}/time_{}_fold_{}_logic_regression_result.csv'.format(project,'process',str(t),str(i)),'w',newline='',encoding='utf8')
                header=['issue','0_prob','1_prob','label']
                file_csv=csv.DictWriter(file,header)
                file_csv.writeheader()
                    
                scores=gb_clf.predict_proba(test_datas)
                file=open('../{}/github_data/result/{}/time_{}_fold_{}_GBDT_result.csv'.format(project,'process',str(t),str(i)),'w',newline='',encoding='utf8')
                header=['issue','0_prob','1_prob','label']
                file_csv=csv.DictWriter(file,header)
                file_csv.writeheader()
                
                for k in range(len(scores)):
                    file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})

def tfidf(words):
    
    dictionary=corpora.Dictionary(words)
    
    corpus = [dictionary.doc2bow(text) for text in words]
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return(dictionary,corpus_tfidf)
                 

def get_tfidf_model(repos):
    
    for repo in repos:
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        for k in range(5,10):
            
            words=[]
            #print(i,len(train_datas),len(train_datas[0]),len(train_labels))
            for j in range(k):
                file_path='../{}/github_data/word_embedding/fold_{}_dimension_{}_issue_labels.csv'.format(project,str(j),str(50))
                file_data=pd.read_csv(file_path)
                issue_label=dict(zip(file_data['issue'],file_data['label']))
                
                for issue,label in issue_label.items():
                    issue_number=issue.split('#')[1]
                    issues=json.load(open('../{}/github_data/issues/{}_info.json'.format(project,issue_number),'r',encoding='utf8'))
                    
                    title=issues['title']
                    title_words=get_token(title.lower())
                    body=issues['body']
                    if(not body):
                        body=''
                    body_words=get_token(body.lower())
                    title_words.extend(body_words)
                    words.append(title_words)
            
            dictionary,tfidf_vec=tfidf(words)
            
            #np.savetxt('../{}/github_data/tfidf/feature.csv'.format(project), tfidf_vec, delimiter = ',')
            with open('../{}/github_data/tfidf/fold_{}_features.json'.format(project,str(k)),'w',encoding='utf8') as file:
                json.dump(list(tfidf_vec),file,indent=4)
                file.close()
            with open('../{}/github_data/tfidf/fold_{}_issue_texts.json'.format(project,str(k)),'w',encoding='utf8') as file:
                json.dump(words,file,indent=4)
                file.close()
            features=[]
            for i in range(len(dictionary)):
                features.append(dictionary[i])
            with open('../{}/github_data/tfidf/fold_{}_feature_names.json'.format(project,str(k)),'w',encoding='utf8') as file:
                json.dump(features,file,indent=4)
                file.close()
            print(repo,k)

def compute_word_tfidf(repos):
    
    for repo in repos:
        project=repo.split('/')[1]
        for k in range(5,10):
            file=open('../{}/github_data/tfidf/fold_{}_word_tf_tfidf.csv'.format(project,str(k)),'w',newline='',encoding='utf8')
            header=['word','cpc_tf','cpc_tfidf']
            file_csv=csv.DictWriter(file,header)
            file_csv.writeheader()
            
            total_issues=[]
            issue_types={}
            with open('../{}/github_data/total_issues.csv'.format(project),'r',encoding='utf8') as file:
                reader=csv.DictReader(file)
                for row in reader:
                    total_issues.append(row['issue'])
                    issue_types[row['issue']]=row['type']
                file.close()
            issue_labels={}
            with open('../{}/github_data/add_indirect_issue_labels.csv'.format(project),'r',encoding='utf8') as file:
                reader=csv.DictReader(file)
                for row in reader:
                    issue_labels[row['issue']]=row['label']
                file.close()
            with open('../{}/github_data/tfidf/fold_{}_feature_names.json'.format(project,str(k)),'r',encoding='utf8') as file:
                feature_names=json.load(file)
                file.close()
            with open('../{}/github_data/tfidf/fold_{}_features.json'.format(project,str(k)),'r',encoding='utf8') as file:
                features=json.load(file)
                file.close()
            with open('../{}/github_data/tfidf/fold_{}_issue_texts.json'.format(project,str(k)),'r',encoding='utf8') as file:
                words=json.load(file)
                file.close()
            dictionary={}
            
            for i in range(len(feature_names)):
                dictionary[i]=feature_names[i]
            cpc_word_num={}
            cpc_word_tfidf={}
            for i in range(len(words)):
                issue=total_issues[i]
                
                if(issue_types[issue]=='pull'):
                    continue
                print(repo,i)
                for word in words[i]:
                    if(word in cpc_word_num.keys()):
                        cpc_word_num[word]=cpc_word_num[word]+1
                    else:
                        cpc_word_num[word]=1
                for key,value in features[i]:
                    if dictionary[key] in cpc_word_tfidf.keys():
                        cpc_word_tfidf[dictionary[key]]=cpc_word_tfidf[dictionary[key]]+value
                    else:
                        cpc_word_tfidf[dictionary[key]]=value
                    
            for word in feature_names:
                print(word)
                if(word in cpc_word_num.keys()):
                    cpc_tf=cpc_word_num[word]  
                else:
                    cpc_tf=0
                    
                if(word in cpc_word_tfidf.keys()):
                    cpc_tfidf=cpc_word_tfidf[word]/cpc_word_num[word]
                else:
                    cpc_tfidf=0
                
                
                file_csv.writerow({'word':word,'cpc_tf':cpc_tf,'cpc_tfidf':cpc_tfidf})
        
def get_tfidf_features(repos,flag):
    
    for repo in repos:
        project=repo.split('/')[1]
        for p in range(5,10):
            with open('../{}/github_data/tfidf/fold_{}_{}feature_names.json'.format(project,str(p),flag),'r',encoding='utf8') as file:
                feature_names=json.load(file)
                file.close()
            with open('../{}/github_data/tfidf/fold_{}_{}features.json'.format(project,str(p),flag),'r',encoding='utf8') as file:
                features=json.load(file)
                file.close()
                
            dictionary={}
            word_index={}
            for i in range(len(feature_names)):
                dictionary[i]=feature_names[i]
                word_index[feature_names[i]]=i
            
            word_cpc_tfidf={}
            with open('../{}/github_data/tfidf/fold_{}_{}word_tf_tfidf.csv'.format(project,str(p),flag),'r',encoding='utf8') as file:
                reader=csv.DictReader(_.replace('\0','') for _ in file)
                for row in reader:
                    if('NUL' in row):
                        continue
                    word_cpc_tfidf[row['word']]=float(row['cpc_tfidf'])
                file.close()
            
            
            new_word_cpc_tfidf=sorted(word_cpc_tfidf.items(),key=lambda x:x[1],reverse=True)
            
            for k in range(50,501,50):
                print(repo,p,k)
                words=[]
                for i in range(k):
                    #print(word_cpc_tfidf[i][0])
                    words.append(word_index[new_word_cpc_tfidf[i][0]])
                tfidf_feature=np.zeros((len(features),k))
                
                for i in range(len(features)):
                    #for j in range(len(words)):
                    for key,value in features[i]:
                        if(key in words):
                            index=words.index(key)
                            tfidf_feature[i][index]=value
                np.savetxt('../{}/github_data/tfidf/fold_{}_{}{}_dimension_features.csv'.format(project,str(p),str(k),flag), tfidf_feature, delimiter = ',')
                
            
                #print(i,len(train_datas),len(train_datas[0]),len(train_labels))
                file_path='../{}/github_data/word_embedding/fold_{}_issue_labels.csv'.format(project,str(p))
                file_data=pd.read_csv(file_path)
                issue_label=dict(zip(file_data['issue'],file_data['label']))
                
                test_datas=[]    
                for issue,label in issue_label.items():
                    issue_number=issue.split('#')[1]
                    issues=json.load(open('../{}/github_data/issues/{}_info.json'.format(project,issue_number),'r',encoding='utf8'))
                        
                    title=issues['title']
                    title_words=get_token(title.lower())
                    body=issues['body']
                    if(not body):
                        body=''
                    body_words=get_token(body.lower())
                    title_words.extend(body_words)
                    temp=[]
                    for word in words:
                        if(dictionary[word] in title_words):
                            temp.append(word_cpc_tfidf[dictionary[word]])
                        else:
                            temp.append(0)
                    test_datas.append(temp)
                np.savetxt('../{}/github_data/tfidf/fold_{}_{}{}_dimension_test_features.csv'.format(project,str(p),str(k),flag), np.array(test_datas), delimiter = ',')
            

def get_word_embedding_model(repos):
    for repo in repos:
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        for i in range(5,10):
            
            words=[]
           
            #print(i,len(train_datas),len(train_datas[0]),len(train_labels))
            for j in range(i):
                file_path='../{}/github_data/word_embedding/fold_{}_issue_labels.csv'.format(project,str(j))
                file_data=pd.read_csv(file_path)
                issue_label=dict(zip(file_data['issue'],file_data['label']))
                
                for issue,label in issue_label.items():
                    issue_number=issue.split('#')[1]
                    issues=json.load(open('../{}/github_data/issues/{}_info.json'.format(project,issue_number),'r',encoding='utf8'))
                    
                    title=issues['title']
                    title_words=get_token(title.lower())
                    body=issues['body']
                    if(not body):
                        body=''
                    body_words=get_token(body.lower())
                    title_words.extend(body_words)
                    words.append(title_words)
                    
            for k in range(50,601,50):
                print(repo,i,str(k))
                #s=save_path+str(k)+'_dimension.model'
                model = gensim.models.Word2Vec(words, sg = 1, window = 5, alpha = 0.025, vector_size = k)
                model.save('../{}/github_data/word_embedding/fold_{}_{}_dimnesion.model'.format(project,str(i),str(k)))
                features=[]
                for word in words:
                    vec=np.zeros((k,))
                    count=0
                    for w in word:
                        if(w in model.wv):
                            vec+=model.wv[w]
                            count+=1
                    if(count==0):
                        features.append(vec)
                    else:
                        features.append(vec/count)
                np.savetxt('../{}/github_data/word_embedding/fold_{}_{}_dimnesion_features.csv'.format(project,str(i),str(k)),features,delimiter=',')
                
                file_path='../{}/github_data/word_embedding/fold_{}_issue_labels.csv'.format(project,str(i))
                file_data=pd.read_csv(file_path)
                issue_label=dict(zip(file_data['issue'],file_data['label']))
                
                test_features=[]
                for issue,label in issue_label.items():
                    issue_number=issue.split('#')[1]
                    issues=json.load(open('../{}/github_data/issues/{}_info.json'.format(project,issue_number),'r',encoding='utf8'))
                    
                    title=issues['title']
                    title_words=get_token(title.lower())
                    body=issues['body']
                    if(not body):
                        body=''
                    body_words=get_token(body.lower())
                    title_words.extend(body_words)
                    
                    vec=np.zeros((k,))
                    count=0
                    for w in title_words:
                        if(w in model.wv):
                            vec+=model.wv[w]
                            count+=1
                    if(count==0):
                        test_features.append(vec)
                    else:
                        test_features.append(vec/count)
                np.savetxt('../{}/github_data/word_embedding/fold_{}_{}_dimnesion_test_features.csv'.format(project,str(i),str(k)),test_features,delimiter=',')
                
                if os.path.exists('../{}/github_data/word_embedding/fold_{}_orig_{}_dimnesion_features.csv'.format(project,str(i),str(k))):
                    os.remove('../{}/github_data/word_embedding/fold_{}_orig_{}_dimnesion_features.csv'.format(project,str(i),str(k)))                 


def train_mix_model(repos,models):
    for repo in repos:
        project=repo.split('/')[1]
        process_datas={}
        with open('../{}/github_data/issue_process_metrics.csv'.format(project),'r',encoding='utf8') as file:
            reader=csv.reader(file)
            for row in reader:
                temp=[row[1],row[3],row[5],row[7],row[9],row[14],row[16],row[17],row[18],row[19],row[20]]
                #temp.extend()
                process_datas[row[0]]=temp
            file.close()
        
        for i in range(5,10):
            train_datas=[]
            train_labels=[]
            datas=[]
            for j in range(i):
                file_path='../{}/github_data/word_embedding/fold_{}_issue_labels.csv'.format(project,str(j))
                file_data=pd.read_csv(file_path)
                issue_labels=list(file_data['label'])
                fold_issues=list(file_data['issue'])
                train_labels.extend(issue_labels)
                
                #datas=[]
                for k in range(len(fold_issues)):
                    datas.append(process_datas[fold_issues[k]])
                
            if('process' in models):
                train_datas=np.array(datas)
                for model in models:
                    if(model=='process'):
                        continue
                    if(model=='tfidf'):
                        train_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimension_features.csv'.format(project,model,str(i),str(50)),delimiter=',')
                        train_datas=np.concatenate((train_datas,train_data),axis=1)
                    if(model=='word_embedding'):
                        train_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimnesion_features.csv'.format(project,model,str(i),str(400)),delimiter=',')
                        train_datas=np.concatenate((train_datas,train_data),axis=1)
            else:
                tfidf_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimension_features.csv'.format(project,'tfidf',str(i),str(50)),delimiter=',')
                train_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimnesion_features.csv'.format(project,'word_embedding',str(i),str(400)),delimiter=',')
                train_datas=np.concatenate((tfidf_data,train_data),axis=1)
            #train_datas.extend(datas.tolist())
                
            print(project,i,len(train_datas),len(train_datas[0]),len(train_labels))
            
            file_path='../{}/github_data/word_embedding/fold_{}_issue_labels.csv'.format(project,str(i))
            file_data=pd.read_csv(file_path)
            test_labels=list(file_data['label'])
            test_issues=list(file_data['issue'])
            
            if('process' in models):
                datas=[]
                for k in range(len(test_issues)):
                    datas.append(process_datas[test_issues[k]])
                test_datas=np.array(datas)
                print(test_datas.shape)
                for model in models:
                    if(model=='process'):
                        continue
                    if(model=='tfidf'):
                        train_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimension_test_features.csv'.format(project,model,str(i),str(50)),delimiter=',')
                        test_datas=np.concatenate((test_datas,train_data),axis=1)
                    if(model=='word_embedding'):
                        train_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimnesion_test_features.csv'.format(project,model,str(i),str(400)),delimiter=',')
                        test_datas=np.concatenate((test_datas,train_data),axis=1)
            else:
                tfidf_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimension_test_features.csv'.format(project,'tfidf',str(i),str(50)),delimiter=',')
                train_data=np.loadtxt('../{}/github_data/{}/fold_{}_{}_dimnesion_test_features.csv'.format(project,'word_embedding',str(i),str(400)),delimiter=',')
                test_datas=np.concatenate((tfidf_data,train_data),axis=1)
                print(test_datas.shape)
               
            print(i,len(test_datas),len(test_datas[0]),len(test_labels))
            
            mix_models=[]
            for model in models:
                mix_models.append(model[0])
            flag='+'.join(mix_models)
            for t in range(10):
                   #svm_clf = svm.SVC(kernel='rbf', C=1,probability=True)
                   #dt_clf=tree.DecisionTreeClassifier()
                   rf_clf=RandomForestClassifier(n_estimators=100)
                   lr_clf=LogisticRegression()
                   #nb_clf= GaussianNB()
                   gb_clf=GradientBoostingClassifier()
                   
                   #svm_clf.fit(train_datas,train_labels)
                   #dt_clf.fit(train_datas,train_labels)
                   rf_clf.fit(train_datas,train_labels)
                   lr_clf.fit(train_datas,train_labels)
                   #nb_clf.fit(train_datas,train_labels)
                   gb_clf.fit(train_datas,train_labels)
                   
                   print(t)
                   '''
                   scores=svm_clf.predict_proba(test_datas)
                   file=open('../{}/github_data/result/{}/time_{}_fold_{}_svm_result.csv'.format(project,flag,str(t),str(i)),'w',newline='',encoding='utf8')
                   header=['issue','0_prob','1_prob','label']
                   file_csv=csv.DictWriter(file,header)
                   file_csv.writeheader()
                   #print(save_path+'fold_'+str(i)+'_'+'svm.csv')
                   for k in range(len(scores)):
                       #print('rf',issues[i])
                       file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})
                   
                   scores=dt_clf.predict_proba(test_datas)
                   file=open('../{}/github_data/result/{}/time_{}_fold_{}_decision_tree_result.csv'.format(project,flag,str(t),str(i)),'w',newline='',encoding='utf8')
                   header=['issue','0_prob','1_prob','label']
                   file_csv=csv.DictWriter(file,header)
                   file_csv.writeheader()
                   
                   for k in range(len(scores)):
                       #print('rf',issues[i])
                       file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})
                       
                   scores=nb_clf.predict_proba(test_datas)
                   file=open('../{}/github_data/result/{}/time_{}_fold_{}_navie_bayes_reuslt.csv'.format(project,flag,str(t),str(i)),'w',newline='',encoding='utf8')
                   header=['issue','0_prob','1_prob','label']
                   file_csv=csv.DictWriter(file,header)
                   file_csv.writeheader()
                   
                   for k in range(len(scores)):
                       #print('rf',issues[i])
                       file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})
                       
                       
                   '''
                   scores=rf_clf.predict_proba(test_datas)
                   file=open('../{}/github_data/result/{}/time_{}_fold_{}_rondom_forest_result.csv'.format(project,flag,str(t),str(i)),'w',newline='',encoding='utf8')
                   header=['issue','0_prob','1_prob','label']
                   file_csv=csv.DictWriter(file,header)
                   file_csv.writeheader()
                   
                   for k in range(len(scores)):
                       #print('rf',issues[i])
                       file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})
                   scores=lr_clf.predict_proba(test_datas)
                   file=open('../{}/github_data/result/{}/time_{}_fold_{}_logic_regression_result.csv'.format(project,flag,str(t),str(i)),'w',newline='',encoding='utf8')
                   header=['issue','0_prob','1_prob','label']
                   file_csv=csv.DictWriter(file,header)
                   file_csv.writeheader()
                   
                   for k in range(len(scores)):
                       #print('rf',issues[i])
                       file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})
                       
                   
                   scores=gb_clf.predict_proba(test_datas)
                   
                   file=open('../{}/github_data/result/{}/time_{}_fold_{}_GBDT_result.csv'.format(project,flag,str(t),str(i)),'w',newline='',encoding='utf8')
                   header=['issue','0_prob','1_prob','label']
                   file_csv=csv.DictWriter(file,header)
                   file_csv.writeheader()
                   
                   for k in range(len(scores)):
                       #print('rf',issues[i])
                       file_csv.writerow({'issue':test_issues[k],'0_prob':scores[k][0],'1_prob':scores[k][1],'label':test_labels[k]})

def get_metric(path,threshold,flag):
    
    print('threshold',threshold)
    labels=[]
    #1_prob=[]
    pre_labels=[]
    
    with open(path) as file:
        reader=csv.DictReader(file)
        for row in reader:
            labels.append(int(row['label']))
            #1_prob.append(float(row['1_prob']))
            if(float(row['1_prob'])>=threshold):
                pre_labels.append(1)
            else:
                pre_labels.append(0)
            
        file.close()
    
    f1=f1_score(labels,pre_labels)
    print('f1',f1)
    
    if(flag):
        #计算MCC,阈值太小，分母为0
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(labels)):
            if(labels[i]==1 and pre_labels[i]==1):
                tp=tp+1
            if(labels[i]==1 and pre_labels[i]==0):
                fn=fn+1
            if(labels[i]==0 and pre_labels[i]==1):
                fp=fp+1
            if(labels[i]==0 and pre_labels[i]==0):
                tn=tn+1
        #print(tn)
        if((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)):
            
            mcc=(tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        else:
            mcc=0
        print('mcc',mcc)
        return(f1,mcc)
    else:
        return(f1)

def get_results(repos,flag):
    
    file=open('../{}/github_data/result/{}_result.csv'.format('astropy',flag),'w',newline='',encoding='utf8')
    header=['project','fold','time','classifier','metric','value']
    file_csv=csv.writer(file)
    file_csv.writerow(header)
    
    classifiers=['GBDT','logic_regression','random_forest']
    for repo in repos:
        project=repo.split('/')[1]
        for time in range(10):
            for fold in range(5,10):
                for classer in classifiers:
                    probs=[]
                    if(classer=='navie_bayes'):
                        result_path='../{}/github_data/result/{}/time_{}_fold_{}_{}_reuslt.csv'.format(project,flag,str(time),str(fold),classer)
                    else:
                        result_path='../{}/github_data/result/{}/time_{}_fold_{}_{}_result.csv'.format(project,flag,str(time),str(fold),classer)
                    print(result_path)
                    with open(result_path,'r',encoding='utf8') as file:
                        reader=csv.DictReader(file)
                        for row in reader:
                            probs.append(float(row['1_prob']))
                        file.close()
                    probs=sorted(probs,reverse=True)
                    threshold=np.percentile(probs, 50)
                    #threshold=0.5
                    f1,mcc=get_metric(result_path,threshold,True)
                    
                    file_csv.writerow([project,fold,time,classer,'f1',f1])
                    file_csv.writerow([project,fold,time,classer,'mcc',mcc])


def compare_project(repos):
    
    file=open('../{}/compare_project.csv'.format('astropy'),'w',newline='',encoding='utf8')
    header=['project','classifier','metric','value']
    file_csv=csv.writer(file)
    file_csv.writerow(header)
    
    for repo in repos:
        print(repo)
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        class_num=0
        method_num=0
        #module_num=0
        
        with open('../{}/{}.csv'.format(project,project),'r',encoding='utf8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                if(row['CountDeclClass']!=''):
                    class_num+=int(row['CountDeclClass'])
                if(row['CountDeclFunction']!=''):
                    method_num+=int(row['CountDeclFunction'])
                #if(row['CountDeclModule']!=''):
                #    module_num+=int(row['CountDeclModule'])
            file.close()
        print(class_num)
        
        file=open('../{}/all_project_info.txt'.format('astropy'),'r')
        for line in file.readlines():
            lines=line.strip().split(',')
            
            if(lines[0]!=project):
                continue
            print(lines)
            ac=float(lines[1])
            mc=float(lines[2])
            mn=float(lines[3])
            cp=float(lines[4])
            sc=float(lines[5])
            se=float(lines[6])
            clcd=float(lines[7])
            clce=float(lines[8])
            file_csv.writerow([project,'-','AvgCyclomatic',float(lines[1])])
            file_csv.writerow([project,'-','MaxCyalomatic',float(lines[2])])
            file_csv.writerow([project,'-','MaxNesting',float(lines[3])])
            file_csv.writerow([project,'-','CountPath',float(lines[4])])
            file_csv.writerow([project,'-','SumCyclomatic',float(lines[5])])
            file_csv.writerow([project,'-','SumEssential',float(lines[6])])
            file_csv.writerow([project,'-','CountLineCodeDecl',float(lines[7])])
            file_csv.writerow([project,'-','CountLineCodeExe',float(lines[8])])
        
        file=open('../{}/{}.txt'.format(project,project),'r')
        for line in file.readlines():
            lines=line.split()
            if(lines[0]=='Project'):
                continue
            file_csv.writerow([project,'-','_'.join(lines[:-1]),float(lines[-1].replace(',', ''))])
            
        #break
        a_depends=0
        f_depends=0
        c_depends=0
        a_count=0
        f_count=0
        c_count=0
        with open('../{}/{}_ArchDependencies.csv'.format(project,project),'r') as file:
            reader=csv.DictReader(file)
            for row in reader:
                a_depends+=1
            file.close()
        
        with open('../{}/{}_ClassDependencies.csv'.format(project,project),'r') as file:
            reader=csv.DictReader(file)
            for row in reader:
                c_depends+=1 
            file.close()
        
        with open('../{}/{}_FileDependencies.csv'.format(project,project),'r') as file:
            reader=csv.DictReader(file)
            for row in reader:
                f_depends+=1 
            file.close()  
        file_csv.writerow([project,'-','ArchDependencies',a_depends])
        file_csv.writerow([project,'-','ClassDependencies',c_depends])
        file_csv.writerow([project,'-','FileDependencies',f_depends])
                
        metrics=['package_count','package_count_set','filter_package_count','filter_package_count_set','java_import_count','java_import_count_set','java_package_count','java_package_count_set','java_filter_package_count','java_filter_package_count_set',
                'python_import_count','python_import_count_set','python_package_count','python_package_count_set','python_filter_package_count','python_filter_package_count_set','c_import_count','c_import_count_set','c_package_count','c_package_count_set','c_filter_package_count','c_filter_package_count_set']
        
        for metric in metrics:
            
            with open('../{}/project_import_info.csv'.format('astropy'),'r') as file:
                reader=csv.DictReader(file)
                for row in reader:
                    if(row['project']==project):
                        file_csv.writerow([project,'-',metric,row[metric]])
                file.close()
        
        with open('../{}/github_data/after_all_repo_issue_ratio.csv'.format('astropy'),'r') as file:
            reader=csv.DictReader(file)
            for row in reader:
                if(row['repo']==repo):
                    file_csv.writerow([project,'-','CPC_Ratio',row['cpc_ratio']])
        
        file=open('../astropy/project_require.txt','r')
        for line in file.readlines():
            lines=line.strip().split(',')
            if(lines[0]=='project' or lines[0]!=project):
                continue
            file_csv.writerow([project,'-','Require',float(lines[1])])
            file_csv.writerow([project,'-','Require_Count',float(lines[2])])
                    
        
def get_import_dependency(repos):
    
    file=open('../{}/project_import_info.csv'.format('astropy'),'w',newline='',encoding='utf8')
    header=['project','java_import_count','java_import_count_set','java_filter_package_count','java_filter_package_count_set',
            'python_import_count','python_import_count_set','python_filter_package_count','python_filter_package_count_set','c_import_count','c_import_count_set','c_filter_package_count','c_filter_package_count_set']
    file_csv=csv.writer(file)
    file_csv.writerow(header)
    
    #repos=['eclipse/deeplearning4j']
    for repo in repos:
        print(repo)
        user=repo.split('/')[0]
        project=repo.split('/')[1]
        
        python_packages=[]
        java_packages=[]
        c_packages=[]
        for root,dirs,files in os.walk('../source_code/{}/'.format(project)):
            
            for file in files:
                if(file.split('.')[-1]=='py'):
                    f=open(os.path.join(root,file),'r',encoding='utf8',errors='ignore')
                    for line in f.readlines():
                        lines=line.strip().split()
                        if(len(lines)==0):
                            continue
                        if(lines[0]=='import' or lines[0]=='from'):
                            python_packages.append(lines[1].split('.')[0])
                elif(file.split('.')[-1]=='java'):
                    if(not os.path.exists(os.path.join(root,file))):
                        continue
                    f=open(os.path.join(root,file),'r',encoding='utf8',errors='ignore')
                    for line in f.readlines():
                        lines=line.strip().split()
                        #print(file,lines)
                        if(len(lines)==0):
                            continue
                        if(lines[0]=='import' or lines[0]=='from'):
                            java_packages.append(lines[1])
                elif(file.split('.')[-1] in ['c','cxx','cpp','cc']):
                    f=open(os.path.join(root,file),'r',encoding='utf8',errors='ignore')
                    for line in f.readlines():
                        lines=line.strip().split()
                        if(len(lines)==0):
                            continue
                        if(lines[0]=='#include'):
                            c_packages.append(lines[1].split('/')[0])
        other_java_packages=[]
        for package in java_packages:            
            package_names='.'.join(package.split('.')[:-1])
            if(package_names==''):
                continue
            if(package.split('.')[0]=='java'):
                continue
            other_java_packages.append(package_names)
        

        other_python_packages=[]
        for package in python_packages:
            if(package==''):
                continue
            if(package==project):
                continue
            other_python_packages.append(package)
        
        
        other_c_packages=[]
        for package in c_packages:
            if(package==''):
                continue
            if('<' in package):
                continue
            other_c_packages.append(package)
        
        #print(other_c_packages)
        file_csv.writerow([project,len(java_packages),len(set(java_packages)),len(other_java_packages),len(set(other_java_packages)),len(python_packages),len(set(python_packages)),len(other_python_packages),len(set(other_python_packages)),
                           len(c_packages),len(set(c_packages)),len(other_c_packages),len(set(other_c_packages))])
        #break
        

if __name__=='__main__':
    
    repos=['astropy/astropy','ipython/ipython','matplotlib/matplotlib','numpy/numpy','pandas-dev/pandas','scikit-learn/scikit-learn','scipy/scipy','microsoft/CNTK','eclipse/deeplearning4j','apache/incubator-mxnet','keras-team/keras','nltk/nltk','pytorch/pytorch','tflearn/tflearn','Theano/Theano','torch/torch7']
    
    ### get repo, issues, comment, and commit from github.
    #get_repo(repos)
    #get_issues(repos)
    #get_comment(repos)
    #get_commit(repos)
    
    ### get original cross project correlated issues
    #get_cpc_issues(repos)
    ### get indirect cross project correlated issues
    #get_new_cpc_issues(repos)
    
    ### get issue metrics
    #get_issue_text_metric(repos)
    #get_issue_famility(repos)
    #get_issue_experience(repos)
    #combine_process_features(repos)
    
    ### get text features based on TF-IDF
    #get_tfidf_model(repos)
    #compute_word_tfidf(repos)
    #get_tfidf_features(repos,'')
    
    ### get text features based on Word Embedding
    #get_word_embedding_model(repos)
    
    ### training model
    #train_process_model(repos)
    #train_mix_model(repos,['process','tfidf'])
    #train_mix_model(repos,['process','word_embedding'])
    #train_mix_model(repos,['process','tfidf','word_embedding'])
    #train_mix_model(repos,['tfidf','word_embedding'])
    #get_results(repos,'process')
    #get_results(repos,'p+t')
    #get_results(repos,'p+w')
    #get_results(repos,'p+t+w')
    #get_results(repos,'t+w')

    ### get project metrics
    #compare_project(repos)
    #get_import_dependency(repos)