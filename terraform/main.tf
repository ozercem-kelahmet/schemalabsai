terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_disk" "boot_disk" {
  name  = "schemalabsai-prod"
  zone  = var.zone
  size  = 500
  type  = "pd-ssd"

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [image, labels, licenses]
  }
}

resource "google_compute_instance" "schemalabsai" {
  name         = "schemalabsai-prod-gpu001"
  machine_type = "n1-standard-8"
  zone         = var.zone

  tags = ["http-server", "https-server"]

  boot_disk {
    source      = google_compute_disk.boot_disk.self_link
    auto_delete = false
  }

  network_interface {
    network    = "default"
    subnetwork = "default"
    access_config {
      nat_ip       = "34.9.180.204"
      network_tier = "PREMIUM"
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  metadata = {
    "enable-osconfig"    = "TRUE"
    "serial-port-enable" = "TRUE"
  }

  service_account {
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [metadata["ssh-keys"], labels]
  }
}

resource "google_compute_firewall" "allow_http" {
  name    = "default-allow-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

resource "google_compute_firewall" "allow_https" {
  name    = "default-allow-https"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["https-server"]
}

resource "google_compute_firewall" "allow_app_ports" {
  name    = "schemalabs-allow-app-ports"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["3000", "5432", "6000", "6379", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}
