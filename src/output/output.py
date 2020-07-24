def output_figure(format, path, name, df=None, fig=None):
    if format == "latex":
        print(df.to_latex(index=False))
    if format == "pdf":
        fig.savefig(path + '%s.pdf' % name)
    if format == "png":
        fig.savefig(path + '%s.png' % name)
    if format == "svg":
        fig.savefig(path + '%s.svg' % name)