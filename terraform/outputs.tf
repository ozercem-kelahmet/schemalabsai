output "instance_name" {
  value = google_compute_instance.schemalabsai.name
}

output "external_ip" {
  value = "34.9.180.204"
}

output "machine_type" {
  value = google_compute_instance.schemalabsai.machine_type
}

output "gpu" {
  value = google_compute_instance.schemalabsai.guest_accelerator[0].type
}
