import os, sys
from rf2steps import RandomForestModel
import rflearning

def make_clf_grid_report(classifier, col2fit, rep_label = 'test'):
    """
    Creates a report of the hyperparameter search
    """
    max_depths = [9,12,13,15,18,22,26,30,40]
    nestimators = [30, 50, 70, 80, 100, 150, 200, 250, 300, 400, 600]
    #max_depths = [8,20,30]
    #nestimators = [10, 20, 30]

    
    ## Report info
    report_dir = os.path.join('reports/',rep_label)
    if not os.path.exists(report_dir):
        print 'creating report directory: %s'%report_dir
        os.mkdir(report_dir)
    reportname = os.path.join(report_dir, 'clf_grid_report.txt')
    
    if os.path.exists(reportname):
        if raw_input('Report with this name already exists...\
            overwrite?(y/n)') not in ('y', 'Y', 'Yes', 'YES', 'yes'):
            print '\n\nAborting!\n\n'
            sys.exit(1)
    
    print 'making classifier grid report with {}'.format(classifier)
    
    ## Writing the paramenters to the report
    frep = open(reportname, 'w')
    frep.write('\ndf_file=%s\n'%classifier)
    frep.write('\ncolumns=%s\n'%str(col2fit))
    frep.write('\nmax_depths=\n%s\n'%str(max_depths))
    frep.write('\nn_estimators=\n%s\n'%str(nestimators))

    
    lrn = rflearning.clf_learning(saved_df=classifier)
    report_dic = lrn.grid_search(col2fit, waitNshow=False, nestimators=nestimators, max_depths=max_depths)
    
    frep.write('\ngrid score =\n%s\n'%str(report_dic['grid_score']))

    ## Save figures
    save_type='png'
    report_dic['fig_grid'].savefig(os.path.join(report_dir, 'clf_fig_gridsearch.%s'%save_type))
    report_dic['fig_mdepth'].savefig(os.path.join(report_dir, 'clf_fig_maxdepth.%s'%save_type))
    report_dic['fig_nestim'].savefig(os.path.join(report_dir, 'clf_fig_nestim.%s'%save_type))
    
    frep.close()
    print '\nFinished writing report file:\n%s'%reportname


    
def make_clf_score_report(classifier, col2fit, rep_label = 'test'):
    """
    Creates a report about the classifier
    """
    ## Report info
    report_dir = os.path.join('reports/',rep_label)
    if not os.path.exists(report_dir):
        print 'creating report directory: %s'%report_dir
        os.mkdir(report_dir)
    reportname = os.path.join(report_dir, 'clf__report.txt')

    if os.path.exists(reportname):
        if raw_input('Report with this name already exists...\
            overwrite?(y/n)') not in ('y', 'Y', 'Yes', 'YES', 'yes'):
            print '\n\nAborting!\n\n'
            sys.exit(1)

    ## Some parameters
    maxdepth = 18
    nestimators = 300

    print 'making classifier score report with {}'.format(classifier)
    
    ## Writing the paramenters to the report
    frep = open(reportname, 'w')
    frep.write('\ndf_file=%s\n'%classifier)
    frep.write('\ncolumns=%s\n'%str(col2fit))
    
    #rfmodel = RandomForestModel('Data/train_2013.csv', 1000)## For testing
    rfmodel = RandomForestModel(saved_df = classifier)
    report_dic = rfmodel.fitNscoreClassifier(col2fit, maxdepth, nestimators, waitNshow=False)
    frep.write('\nfeature importances:\n%s\n'%str(report_dic['features_importances']))
    frep.write('\n\nCross validation accuracy (std): %f (%f)\n'%(
        report_dic['scores_mean'], report_dic['scores_std']))
    frep.write('\n ROC area under curve: %f\n'%report_dic['roc_auc'])

    frep.write('\n\n\n\n'+100*'-' + '\n End of report\n')

    ## Save figures
    save_type='png'
    report_dic['fig_importance'].savefig(os.path.join(report_dir, 'clf_fig_importance.%s'%save_type))
    report_dic['fig_prob'].savefig(os.path.join(report_dir, 'clf_fig_prob.%s'%save_type))
    report_dic['fig_roc'].savefig(os.path.join(report_dir, 'clf_fig_roc.%s'%save_type))

    frep.close()
    print '\nFinished writing report file:\n%s'%reportname

if __name__=="__main__":
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD', 'Avg_RhoHV', 'Range_RhoHV'
                ]

    #make_clf_score_report('saved_df/test30k.h5', coltofit, 'test30k')
    make_clf_grid_report('saved_df/test30k.h5', coltofit, 'test30k')
