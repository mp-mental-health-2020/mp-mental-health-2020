def output_latex(df, fig, name, format):
    if format == "latex":
        print(df.to_latex(index=False))
    if format == "pdf":
        fig.savefig('./pictures/%s.pdf' % name)
    if format == "png":
        fig.savefig('./pictures/%s.png' % name)
    if format == "svg":
        fig.savefig('./pictures/%s.svg' % name)