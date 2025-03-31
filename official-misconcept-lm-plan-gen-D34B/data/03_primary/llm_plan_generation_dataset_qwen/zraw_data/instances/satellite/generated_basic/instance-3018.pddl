(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	spectrograph0 - mode
	star3 - direction
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	star5 - direction
	star4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star5)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star4)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite1 star8)
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 spectrograph0)
))

)
