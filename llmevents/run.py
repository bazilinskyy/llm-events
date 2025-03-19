# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import llmevents as llme

llme.logs(show_level='info', show_color=True)
logger = llme.CustomLogger(__name__)  # use custom logger

# const
SAVE_P = True  # save pickle files with data
LOAD_P = False  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = True  # filter output data
CLEAN_DATA = True  # clean output data
PROCESS_ANSWERS = True  # process answers in output
ANALYSE_DATA = True  # analyse output data
SHOW_OUTPUT = True  # should figures be plotted


if __name__ == '__main__':
    # create object for working with heroku data
    reports = llme.common.get_configs('reports')
    llmevents = llme.analysis.LLMEvents(files_reports=reports, save_p=SAVE_P, load_p=LOAD_P, save_csv=SAVE_CSV)
    # read data data
    df = llmevents.read_data(filter_data=FILTER_DATA,
                             clean_data=CLEAN_DATA,
                             analyse_data=ANALYSE_DATA,
                             process_answers=PROCESS_ANSWERS)
    print(df.columns)
    print(df.head)
    # apply categorization
    logger.info('Data from {} reports included in analysis.', df.shape[0])
    if SHOW_OUTPUT:
        # Output
        analysis = llme.analysis.Analysis()
        logger.info('Creating figures.')
        # histograms of questions
        analysis.hist(df, x=['q1_category'],
                      yaxis_title='Q1. Describe the accident',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        analysis.hist(df, x=['q2_category'],
                      yaxis_title='Q2. Involved parties(Who)',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        analysis.hist(df, x=['q3_category'],
                      yaxis_title='Q3. Accident location details(Where)',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        analysis.hist(df, x=['q4_category'],
                      yaxis_title='Q4. Time and environmental conditions(When)',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        analysis.hist(df, x=['q5_category'],
                      yaxis_title='Q5. Accident damage and consequences(What)',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        analysis.hist(df, x=['q6_category'],
                      yaxis_title='Q6. Responsibility and contributing factors(Why)',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        analysis.hist(df, x=['q7_category'],
                      yaxis_title='Q7. Traffic and vehicle behavior(How)',
                      marginal=None,
                      pretty_text=True,
                      save_file=True)
        # some scatter plot
        # analysis.scatter(df, x='report', y='response', color='report', pretty_text=True, save_file=True)
        # # some histogram
        # analysis.hist(df, x=['report'],  pretty_text=True, save_file=True)
        # # some map
        # analysis.map(data, color='', save_file=True)
        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
