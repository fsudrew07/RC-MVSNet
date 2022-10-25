import kfp.components
from kfp.dsl import VolumeOp
from kubernetes.client import V1Toleration
from kfp import compiler
from kfp import dsl

DATA_PATH = "/data"
MINIMUM_PVC_SIZE = 200
KUBEFLOW_GPU_NODE_POOL = "dt-cv-dw-gpu"
KUBEFLOW_TOLERATION = "dt-cv-dw"
KUBEFLOW_CPU_LOW_NODE_POOL = "dt-cv-dw-low-cpu"


def get_pvc(size_in_gb: int) -> VolumeOp:

    return dsl.VolumeOp(
        name="pvc-volume",
        resource_name="pvc-volume",
        storage_class="standard-storage",
        size=f"{size_in_gb}Gi",
        modes=dsl.VOLUME_MODE_RWO,
        set_owner_reference=True,
    )


get_data_from_gcs_pvc_op = """
name: Get Data From GCS
description: Gets data from GCS
inputs:
- {name: input_path, type: String}
- {name: output_path, type: String}
implementation:
  container:
    image: gcr.io/google.com/cloudsdktool/google-cloud-cli:latest
    command:
    - sh
    - -c
    - |
      mkdir -p "$1"
      echo "Copying files from $0"
      gsutil -m cp -r "$0" "$1"
    args:
    - {inputValue: input_path}
    - {inputValue: output_path}
"""

run_evaluate_on_tankntemples = """
name: Run Evaluation
description: Evaluate Tank and temple dataset
inputs:
- {name: output_path, type: String}
- {name: output_path_pc, type: String}
- {name: test_data_path, type: String}
implementation:
  container:
    image: gcr.io/unity-ai-dt-cv-data-lake-test/rc-mvsnet@sha256:b48cb8d60a941c50cb434b9d698fc48c7c9eaf042f6804433224729eb11b3150t
    command:
    - sh
    - -c
    - |
      python eval_rcmvsnet_tanks.py --split "intermediate" --loadckpt "./pretrain/model_000014_cas.ckpt"  --plydir "$1" --outdir "$0" --testpath "$2"
    args:
    - {inputValue: output_path}
    - {inputValue: output_path_pc}
    - {inputValue: test_data_path}
"""


@dsl.pipeline(
    name="Evaluate the model", description="Evaluate the model",
)
def evaluate_the_model(
    docker: str = "gcr.io/unity-ai-dt-cv-data-lake-test/rc-mvsnet@sha256:b48cb8d60a941c50cb434b9d698fc48c7c9eaf042f6804433224729eb11b3150t",
    dataset_gcs_path: str = "raw-dataset-b29d/source=tanks_and_temples"
):

    # Pipeline definition
    vop = get_pvc(size_in_gb=MINIMUM_PVC_SIZE)

    download = kfp.components.load_component_from_text(get_data_from_gcs_pvc_op)
    get_dataset_from_gcs_task = download(
        input_path=f"gs://{dataset_gcs_path}/*", output_path="/mvs/tankandtemples"
    ).add_pvolumes({DATA_PATH: vop.volume})
    get_dataset_from_gcs_task.add_toleration(V1Toleration(key="dedicated", value=KUBEFLOW_TOLERATION))
    get_dataset_from_gcs_task.add_node_selector_constraint(
        label_name="cloud.google.com/gke-nodepool", value=KUBEFLOW_CPU_LOW_NODE_POOL
    )

    evaluate = kfp.components.load_component_from_text(run_evaluate_on_tankntemples)
    run_eval_task = evaluate(output_path="/data", output_path_pc="",
                             test_data_path="/mvs/tankandtemples").add_pvolumes({
        DATA_PATH:get_dataset_from_gcs_task.pvolume})
    run_eval_task.add_toleration(V1Toleration(key="dedicated", value=KUBEFLOW_TOLERATION))
    run_eval_task.add_node_selector_constraint(
        label_name="cloud.google.com/gke-nodepool", value=KUBEFLOW_GPU_NODE_POOL
    )


cmplr = compiler.Compiler()
cmplr.compile(evaluate_the_model, package_path='my_pipeline.yaml')
