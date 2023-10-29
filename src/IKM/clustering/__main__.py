from ikm.clust.ikm import IKM
from ikm.clust.dim_selection import Algorithm
from another_clustering import coeffs_clustering
import os


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

    path = "data/final_files"

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
                                        ikm.ikm_process(path=path,
                                                        z_score=z_sc, box_cox=bc,
                                                        z_normalization=z_norm,
                                                        z_transform_mode=z_transform_mode,
                                                        excl_wm=excl_wm,
                                                        band=band,
                                                        hilbert=hilbert, error=error,
                                                        report_name='report')

                                            # if clust_mtd == 'ikm':
                                            #     ikm.ikm_process(excl_wm =True, path=path, error='hamming')
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
                                        ikm.ikm_process(path=path,
                                                        z_score=z_sc, box_cox=bc,
                                                        z_normalization=z_norm,
                                                        z_transform_mode=z_transform_mode,
                                                        excl_wm=excl_wm,
                                                        band=band,
                                                        hilbert=hilbert, error=error,
                                                        report_name='report')
                                                    
    except Exception:
        pass

        # if clust_mtd == 'ikm':
        #     ikm.ikm_process(excl_wm=True, path=path)
        # elif clust_mtd == 'other':
        #     coeffs_clustering.cluster_coeffs_chosen_el(set_expl=report_name)
        # report_name = ''


def main():
    """
    Entry point for running the IKM.
    """

    ikm = IKM()

    path = "../../../data/normal_summer"
    path_save_file_per_event = "../../../data/current_experiment"

    # errors = ['total','eucl', 'max', 'jaccard']
    errors = ['total','eucl']


    for error in errors:
        print(error)
        ikm.ikm_process(path=path,
                        num_clusters = 3,
                        # excl_wm = [],
                        # specific_windmills=1,

                        z_score=True,
                        z_normalization=True, 
                        box_cox=False, # TODO: Ask Katerina is we should use this transformation as we have negative values.

                        error=error,
                        report_name='report',
                        
                        how_to_process_data = "all",
                        kind_mean="all",
                        tries = 50, #100
                        steps = 25, #75
                        samples_per_file = None, # Number of samples ( nº or None )
                        path_save_file_per_event = path_save_file_per_event) 
        
        # TODO: Borrar break


if __name__ == '__main__':
    main()
