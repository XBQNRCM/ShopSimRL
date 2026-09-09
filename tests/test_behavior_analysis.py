"""State-transition and denominator checks for the retrospective analysis."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('behavior',Path(__file__).parents[1]/'scripts/analyze_behavior.py')
behavior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(behavior)

def episode():
    results={'page_type':'search_results','products':[{'asin':'A','rank':1}]}
    detail={'page_type':'product_detail','product':{'asin':'A'},
            'available_options':{'size':['S','M']},'selected_options':{}}
    selected={**detail,'selected_options':{'size':'S'}}
    changed={**detail,'selected_options':{'size':'M'}}
    transitions=[('search[shoe]',results),('click[A]',detail),('click[size=S]',selected),
                 ('click[size=M]',changed),('click[back to search]',results),('click[A]',detail)]
    steps=[{'step_index':i+1,'action':a,'model':{'usage':{'completion_tokens':10}},
            'environment':{'action_feedback':{'valid':True},'observation_state':state}}
           for i,(a,state) in enumerate(transitions)]
    return {'status':'completed','episode_id':'example','job':{'task_id':1,'sample_id':0},
            'reset':{'observation_state':{'page_type':'search_home'}},'steps':steps,
            'final':{'done':True,'reward':1.,'reward_detail':{'r_success':1}}}

def test_option_refresh_is_not_a_new_product_visit():
    row,_,_=behavior.extract_episode(episode(),cohort='test')
    assert (row['product_opens'],row['unique_products'],row['product_revisits'])==(2,1,1)
    assert (row['option_selections'],row['option_changes'],row['backtracks'])==(2,1,1)

def test_protocol_repair_is_a_model_turn_but_not_an_environment_action():
    trace=episode()
    trace['steps'].append({'step_index':7,'action':None,'environment':None,
                           'model':{'protocol_error':'malformed','usage':{'completion_tokens':30}}})
    row,turns,_=behavior.extract_episode(trace,cohort='test')
    assert (row['model_steps'],row['actions'],row['protocol_errors'])==(7,6,1)
    assert row['tokens_per_turn']==90/7 and turns[-1]['phase']=='protocol_repair'

def test_missing_usage_does_not_silently_shorten_the_denominator():
    trace=episode(); trace['steps'][0]['model']={}
    row,_,_=behavior.extract_episode(trace,cohort='test')
    assert row['tokens_per_turn'] is None and row['token_turns']==5
    assert behavior.describe([row])['token_complete_episodes']==0

def test_technical_episode_is_not_a_zero_reward_failure():
    row,_,_=behavior.extract_episode(episode(),cohort='train',scored_override=False)
    assert row['outcome']=='technical' and row['reward'] is None and row['success'] is None

def test_task_pairing_averages_siblings_before_comparison():
    result=behavior.paired_tasks([{'task_id':1,'x':0},{'task_id':1,'x':2},{'task_id':2,'x':5}],
                                 [{'task_id':1,'x':3},{'task_id':2,'x':5}],['x'])
    assert result['paired_tasks']==2
    assert result['metrics']['x']['difference']==1
