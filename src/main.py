import click
from runs import RunFactory


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='adult_fedavg', help='Run to execute')
@click.option('--project_name', '-p', default='AdultFedAvg', help='Project name')
@click.option('--start_index', '-s', default=51, help='Start index')
@click.option('--metric_name', '-m', default='demographic_parity', help='Metric name')
@click.option('--id', '-i', default='test', help='Run id')
@click.option('--group_name', '-g', default='Gender', help='Sensitive Attribute')
@click.option('--onlyperf', '-o', is_flag=True, help='Monitor only performance')
@click.option('--threshold', '-t', default=0.2, help='Fairness threshold')
@click.option('-metrics_list', '-ml', multiple=True, help='List of metrics')
@click.option('-groups_list', '-gl', multiple=True, help='List of groups')
@click.option('-threshold_list', '-tl', type=float, multiple=True, help='List of threshold')
def main(run, project_name, start_index, metric_name, id,
         group_name, use_hale, onlyperf, threshold,
         metrics_list, groups_list, threshold_list):

    run = RunFactory.create_run(run,
                                project_name=project_name,
                                start_index=start_index,
                                metric_name=metric_name,
                                id=id,
                                group_name=group_name,
                                use_hale=use_hale,
                                onlyperf=onlyperf,
                                threshold=threshold,
                                metrics_list=metrics_list,
                                groups_list=groups_list,
                                threshold_list=threshold_list
                                )
    run()


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main()
