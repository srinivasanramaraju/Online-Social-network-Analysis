"""
sumarize.py
"""
import pickle
import string



def main():
    totno=0
    with open('users.pkl', 'rb') as f:
        users = pickle.load(f)
    
    
    file = open('summary.txt','w') 
    for user in users:
          totno+=len(user['followers'])
    file.write("Number of users collected: %d Followers: %d" %(len(users),totno) )
    
  
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)    
    file.write("\nNumber of messages/tweets collected: %d" % results['messages'] )

    file.write("\nNumber of communities discovered: %d" % results['communities'])    
    file.write("\nAverage number of users per community: %d" %results['average'])
    file.write("\nNumber of instances per class found: Male users %d Female users %d" % (results['malecount'] ,results['femalecount']))
    file.write("\nOne example from each class: ")
    sampletweet=results['male']
    printable = set(string.printable)
    scr_name="".join(list(filter(lambda x: x in printable,sampletweet['user']['screen_name'])))
    name="".join(list(filter(lambda x: x in printable,sampletweet['user']['name'])))
    desc="".join(list(filter(lambda x: x in printable,sampletweet['user']['description'])))
    message="".join(list(filter(lambda x: x in printable,sampletweet['text'])))
  
    file.write('\n Male Tweet: \n\tscreen_name=%s\n\tname=%s\n\tdescr=%s\n\ttext=%s' %
            (scr_name,
             name,
             desc,
             message
             ))
    sampletweet=results['female']  
    scr_name="".join(list(filter(lambda x: x in printable,sampletweet['user']['screen_name'])))
    name="".join(list(filter(lambda x: x in printable,sampletweet['user']['name'])))
    desc="".join(list(filter(lambda x: x in printable,sampletweet['user']['description'])))
    message="".join(list(filter(lambda x: x in printable,sampletweet['text'])))       
    file.write('\n Female Tweet: \n\tscreen_name=%s\n\tname=%s\n\tdescr=%s\n\ttext=%s' %
            (scr_name,
             name,
             desc,
             message
             ))           
if __name__ == '__main__':
    main()  