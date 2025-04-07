# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>, Weihang You <weihangyou@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import llmevents as llme

llme.logs(show_level=llme.common.get_configs("logger_level"), show_color=True)
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
SHOW_OUTPUT_CONTEXT = True  # should figures with keypress data be plotted-
SHOW_OUTPUT_HIST = False  # should figures with stimulus data to be plotted


if __name__ == '__main__':
    # create object for working with heroku data
    reports = llme.common.get_configs('reports')
    llmevents = llme.analysis.LLMEvents(files_reports=reports, save_p=SAVE_P, load_p=LOAD_P, save_csv=SAVE_CSV)
    # read data data
    df = llmevents.read_data(filter_data=FILTER_DATA,
                             clean_data=CLEAN_DATA,
                             analyse_data=ANALYSE_DATA,
                             process_answers=PROCESS_ANSWERS)
    # apply categorization
    logger.info('Data from {} reports included in analysis.', df.shape[0])
    if SHOW_OUTPUT:
        # Output
        analysis = llme.analysis.Analysis()
        logger.info('Creating figures.')
        # contextual analysis
        if SHOW_OUTPUT_CONTEXT:
            analysis.sunburst(df, save_file=True)
            analysis.node_graph(df, save_file=True)
            analysis.sankey(df, save_file=True)
        # histograms of questions
        if SHOW_OUTPUT_HIST:
            analysis.hist(df, x=['q1_category'],
                          yaxis_title='Q1. Describe the accident',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_av_brand'],
                          yaxis_title='Q2. Involved parties(Who) - automated vehicle brand',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_av_model'],
                          yaxis_title='Q2. Involved parties(Who) - automated vehicle model',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_av_year'],
                          yaxis_title='Q2. Involved parties(Who) - automated vehicle year',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_av_mode'],
                          yaxis_title='Q2. Involved parties(Who) - automated vehicle mode',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_other_road_user'],
                          yaxis_title='Q2. Involved parties(Who) - other road user',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_other_vehicle'],
                          yaxis_title='Q2. Involved parties(Who) - other vehicle',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q2_av_mode'],
                          yaxis_title='Q2. Involved parties(Who) - automated vehicle mode',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q3_address'],
                          yaxis_title='Q3. Address',
                          marginal=None,
                          # pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q3_street_type'],
                          yaxis_title='Q3. Street type',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q3_lanes'],
                          yaxis_title='Q3. Lanes',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q3_category'],
                          yaxis_title='Q3. Accident location details(Where)',
                          marginal=None,
                          pretty_text=True,
                          save_file=True)
            analysis.hist(df, x=['q3_area_type'],
                          yaxis_title='Q3. Area type',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q4_weather'],
                          yaxis_title='Q4. Time and environmental conditions(When): Weather',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q4_lighting'],
                          yaxis_title='Q4. Time and environmental conditions(When): Lightning',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q4_surface'],
                          yaxis_title='Q4. Time and environmental conditions(When): Surface',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q4_conditions'],
                          yaxis_title='Q4. Time and environmental conditions(When): Conditions',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q5_collision_type'],
                          yaxis_title='Q5. Collision type',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q5_av_damage'],
                          yaxis_title='Q5. Accident damage and consequences(What): AV damage',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q5_injuries'],
                          yaxis_title='Q5. Accident damage and consequences(What): Injuries',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q6_av_at_fault'],
                          yaxis_title='Q6. Responsibility and contributing factors(Why)',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q6_contributing_factors'],
                          yaxis_title='Q6. Responsibility and contributing factors(Why): Contributing factors',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q7_traffic_conditions'],
                          yaxis_title='Q7. Traffic and vehicle behavior(How): Traffic conditions',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q7_av_movement'],
                          yaxis_title='Q7. Traffic and vehicle behavior(How): AV movement',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q7_other_road_user_movement'],
                          yaxis_title='Q7. Traffic and vehicle behavior(How): Other road user movement',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q7_same_direction'],
                          yaxis_title='Q7. Traffic and vehicle behavior(How): Same direction',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
            analysis.hist(df, x=['q7_same_lane'],
                          yaxis_title='Q7. Traffic and vehicle behavior(How): Same lane',
                          marginal=None,
                          pretty_text=True,
                          save_file=True,
                          open_browser=True)
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
