import click
from runs import RunFactory


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='adult_glofair', help='Run to execute')
@click.option('--project_name', '-p', default='AdultGlofair', help='Project name')
@click.option('--num_clients', '-n', default=10, help='Number of clients')
@click.option('--metric_name', '-m', default='demographic_parity', help='Metric name')
@click.option('--id', '-i', default='adult_glofair', help='Run id')
@click.option('--onlyperf', '-o', is_flag=True, help='Monitor only performance')
@click.option('--use_wandb', '-w', is_flag=True, help='Use wandb logger')
@click.option('--threshold', '-t', default=0.2, help='Fairness threshold')
@click.option('-metrics_list', '-ml', multiple=True, help='List of metrics')
@click.option('-groups_list', '-gl', multiple=True, help='List of groups')
@click.option('-threshold_list', '-tl', type=float, multiple=True, help='List of threshold')
@click.option('--experiment', '-e', default='alpha_09', help='Experiment name')
def main(run, project_name, num_clients, metric_name, id,
         use_wandb, onlyperf, threshold,
         metrics_list, groups_list, threshold_list,experiment):

    run = RunFactory.create_run(run,
                                project_name=project_name,
                                num_clients=num_clients,
                                experiment=experiment,
                                metric_name=metric_name,
                                id=id,
                                use_wandb=use_wandb,
                                onlyperf=onlyperf,
                                threshold=threshold,
                                metrics_list=metrics_list,
                                groups_list=groups_list,
                                threshold_list=threshold_list
                                )
    run()


if __name__ == '__main__':
    main()
