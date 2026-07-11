% ===== 建议插入第4章实验设置 =====
\subsection{通信退化强度设置}
在保持兴趣点分布、空间风险区域、初始位置、能量参数、慢环周期和随机种子不变的条件下，本文设置低、中、高三档通信退化强度。中档采用当前基准通信配置，低档和高档分别按 $0.75$ 和 $1.25$ 的倍率调整距离衰减或局部遮挡导致的丢包、时延均值及随机扰动幅值。对于局部遮挡机制，倍率同时作用于遮挡区域的附加丢包。当前仿真器中的带宽由瞬时丢包率通过 $b_t=10^6(1-\ell_t)$ 计算，因此带宽随丢包退化同步变化，而未单独设置带宽衰减系数。三档链路统计见表~\ref{tab:comm-degradation-link-stats}。

\begin{table}[htbp]
  \centering
  \caption{不同通信退化强度下的链路统计}
  \label{tab:comm-degradation-link-stats}
  \begin{tabular}{llcccc}
    \toprule
    通信机制 & 强度 & 平均丢包率 & 平均时延/s & 平均带宽/Mbps & 弱链路比例 \\
    \midrule
    C1 & 低 & 0.142 & 0.078 & 0.858 & 0.000 \\
    C1 & 中 & 0.189 & 0.105 & 0.811 & 0.000 \\
    C1 & 高 & 0.237 & 0.131 & 0.763 & 0.012 \\
    C2 & 低 & 0.143 & 0.078 & 0.857 & 0.000 \\
    C2 & 中 & 0.191 & 0.105 & 0.809 & 0.001 \\
    C2 & 高 & 0.239 & 0.131 & 0.761 & 0.027 \\
    \bottomrule
  \end{tabular}
\end{table}

% ===== 建议插入第5章实验分析 =====
\subsection{通信退化鲁棒性分析}
图~\ref{fig:comm-scan-architecture} 比较了两类通信退化机制下三种决策结构的有效任务完成率。误差线表示不同随机种子结果的标准差。

在距离衰减主导条件下，CDSL 的 $R_{\mathrm{task}}$ 由 0.270 变化至 0.270，WCDL 的 $R_{\mathrm{task}}$ 由 0.270 变化至 0.270，FDLC 的 $R_{\mathrm{task}}$ 由 0.614 变化至 0.491。FDLC 从低档到高档的变化幅度并未同时小于两种基线，因此该组结果不支持将其概括为对所有退化变化均更稳定。

在局部遮挡主导条件下，CDSL 的 $R_{\mathrm{task}}$ 由 0.270 变化至 0.129，WCDL 的 $R_{\mathrm{task}}$ 由 0.270 变化至 0.104，FDLC 的 $R_{\mathrm{task}}$ 由 0.552 变化至 0.602。FDLC 从低档到高档的下降幅度不大于两种基线，表明其在该退化机制下具有更稳定的任务收益。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\textwidth]{../results/comm_degradation_scan_20260620_103457/fig_architecture_degradation_scan.pdf}
  \caption{不同通信退化强度下的决策结构对比}
  \label{fig:comm-scan-architecture}
\end{figure}

图~\ref{fig:comm-scan-coupling} 给出了局部遮挡主导场景下不同耦合机制的有效任务完成率和平均回传时延。

Periodic Goal 的 $R_{\mathrm{task}}$ 为 0.296/0.142/0.125，$\bar{T}_{\mathrm{ret}}=49.27/52.45/71.89\,\mathrm{s}$；Event-driven Goal 的 $R_{\mathrm{task}}$ 为 0.451/0.339/0.215，$\bar{T}_{\mathrm{ret}}=70.94/94.48/114.70\,\mathrm{s}$；Full Coupling 的 $R_{\mathrm{task}}$ 为 0.552/0.655/0.602，$\bar{T}_{\mathrm{ret}}=6.93/11.00/18.69\,\mathrm{s}$。Full Coupling 在中、高退化条件下均取得最高完成率。时延结果需与完成率联合解释，以避免将更早终止误判为更快回传。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\textwidth]{../results/comm_degradation_scan_20260620_103457/fig_coupling_degradation_scan.pdf}
  \caption{不同通信退化强度下的耦合机制消融结果}
  \label{fig:comm-scan-coupling}
\end{figure}

图~\ref{fig:comm-scan-information} 比较了三种信息利用方式的有效任务完成率、返航终止率和越界终止率。

Comm-only 在低、中、高退化下的 $R_{\mathrm{task}}$ 分别为 0.575、0.683 和 0.644，返航率分别为 0.067、0.067 和 0.333，越界率分别为 0.933、0.933 和 0.667。

Energy-only 在低、中、高退化下的 $R_{\mathrm{task}}$ 分别为 0.270、0.281 和 0.104，返航率分别为 0.000、0.067 和 1.000，越界率分别为 1.000、0.933 和 0.000。

Comm+Energy 在低、中、高退化下的 $R_{\mathrm{task}}$ 分别为 0.533、0.671 和 0.592，返航率分别为 0.600、1.000 和 1.000，越界率分别为 0.400、0.000 和 0.000。

这些指标共同刻画通信收益与任务闭合状态；若完成率提高同时伴随越界率上升，则不能仅依据原始任务收益判断信息利用方式的综合效果。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{../results/comm_degradation_scan_20260620_103457/fig_information_degradation_scan.pdf}
  \caption{不同通信退化强度下的信息利用消融结果}
  \label{fig:comm-scan-information}
\end{figure}

在 E210 能量受限条件下，各信息利用方式的结果如下。

Comm-only 在中、高退化下的 $R_{\mathrm{task}}$ 分别为 0.685 和 0.617，能量终止率分别为 0.000 和 0.400。

Energy-only 在中、高退化下的 $R_{\mathrm{task}}$ 分别为 0.255 和 0.104，能量终止率分别为 1.000 和 1.000。

Comm+Energy 在中、高退化下的 $R_{\mathrm{task}}$ 分别为 0.525 和 0.518，能量终止率分别为 0.867 和 1.000。

现有输出未记录独立的能量约束触发次数，因此此处以能量终止率作为可观测代理。
