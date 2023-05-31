from ikm.clust.ikm import IKM
from ikm.clust.dim_selection import Algorithm
from another_clustering import coeffs_clustering


# clust_mtd can be ikm or other
def diff_el_positions_clust(clust_mtd='ikm'):
    """
    This function runs IKM algorithm on different parts of electrodes.
    Parameters:
    clust_mtd (string): Can be 'ikm' or 'other' for DBSCAN, agglomerative or K-means
    """

    z_score = [False, True]
    boxcox = [False, True]
    z_normalization = [False, True]
    z_transform_mode = [None, 'magnitude', 'phase']
    excl_wm = [False, True]
    band = [None, 'delta', 'theta', 'beta', 'gamma', 'alpha']
    hilbert = [None, 'phase', 'amplitude']
    error = ['eucl', 'total', 'max', 'jaccard', 'hamming']

    loading_setups = ['one_f', 'two_f']
    # paths = ["/media/data/lazarenkom98dm/objects-first-visit-full",
    #          "/media/data/lazarenkom98dm/objects-second-visit-full"]

    paths = ["data/dataset_split_events"]

    ikm = IKM()

    try:

        for z_sc in z_score:
            for bc in boxcox:
                for z_norm in z_normalization:
                    for z_transform_mode in z_transform_mode:
                        for excl_wm in excl_wm:
                            for band in band:
                                for hilbert in hilbert:
                                    for error in error:
                                        for loading_setup in loading_setups:
                                            if loading_setup == 'one_f':
                                                for path in paths:
                                                    ikm.ikm_process(path=path, loading_setup=loading_setup,
                                                                    z_score=z_sc, box_cox=bc,
                                                                    z_normalization=z_norm,
                                                                    z_transform_mode=z_transform_mode,
                                                                    excl_wm=excl_wm,
                                                                    band=band,
                                                                    hilbert=hilbert, error=error,
                                                                    report_name='report')
                                            elif loading_setup == 'two_f':
                                                ikm.ikm_process(path=paths[0], path1=paths[1],
                                                                loading_setup=loading_setup, z_score=z_sc,
                                                                box_cox=bc,
                                                                z_normalization=z_norm,
                                                                z_transform_mode=z_transform_mode,
                                                                excl_wm=excl_wm,
                                                                band=band,
                                                                hilbert=hilbert, error=error,
                                                                report_name='report')

                                            # if clust_mtd == 'ikm':
                                            #
                                            #     ikm.ikm_process(excl_wm =True, path=paths[0], path1=paths[1], loading_setup = 'one_f', error='hamming')
                                            # elif clust_mtd == 'other':
                                            #     coeffs_clustering.cluster_coeffs_chosen_el(set_expl=report_name)
                                            # report_name = ''

        for z_sc in z_score:
            for bc in boxcox:
                for z_norm in z_normalization:
                    for z_transform_mode in z_transform_mode:
                        for excl_wm in excl_wm:
                            for band in band:
                                for hilbert in hilbert:
                                    for error in error:
                                        for loading_setup in loading_setups:
                                            if loading_setup == 'one_f':
                                                for path in paths:
                                                    ikm.ikm_process(path=path, loading_setup=loading_setup,
                                                                    z_score=z_sc, box_cox=bc,
                                                                    z_normalization=z_norm,
                                                                    z_transform_mode=z_transform_mode,
                                                                    excl_wm=excl_wm,
                                                                    band=band,
                                                                    hilbert=hilbert, error=error,
                                                                    report_name='report')
                                            elif loading_setup == 'two_f':
                                                ikm.ikm_process(path=paths[0], path1=paths[1],
                                                                loading_setup=loading_setup, z_score=z_sc,
                                                                box_cox=bc,
                                                                z_normalization=z_norm,
                                                                z_transform_mode=z_transform_mode,
                                                                excl_wm=excl_wm,
                                                                band=band,
                                                                hilbert=hilbert, error=error,
                                                                report_name='report')
                                                    
    except Exception:
        pass

        # if clust_mtd == 'ikm':
        #
        #     ikm.ikm_process(excl_wm=True, path=paths[0], path1=paths[1], loading_setup='two_f')
        # elif clust_mtd == 'other':
        #     coeffs_clustering.cluster_coeffs_chosen_el(set_expl=report_name)
        # report_name = ''


def main():
    """
    Entry point for running the IKM.
    """

    ikm = IKM()

    # paths = ["/media/data/lazarenkom98dm/objects-first-visit-full",
    #          "/media/data/lazarenkom98dm/objects-second-visit-full"]
    paths = ["../../../data/dataset_split_events"]
    loading_setup = "one_f"
    num_clusters = 2
    # hilbert = "amplitude" / "phase"

    errors = ['total', 'max', 'jaccard', 'hamming']

    # for error in errors:
    #     ikm.ikm_process(path=paths[0], 
    #                     path1=paths[1], 
    #                     loading_setup="two_f",
    #                     band='delta', 
    #                     num_clusters = num_clusters, 
    #                     error=error,
    #                     report_name='report')
        
    for error in errors:
        ikm.ikm_process(path=paths[0],
                        # path1=paths[1], 
                        loading_setup=loading_setup,
                        band='delta', 
                        hilbert='amplitude',
                        num_clusters = num_clusters,
                        # excl_wm = [],
                        # specific_windmills=1,

                        z_score=True,
                        z_normalization=True, 
                        box_cox=True,
                        z_transform_mode = "magnitude",

                        error=error,
                        report_name='report')
        
        break


if __name__ == '__main__':
    main()
