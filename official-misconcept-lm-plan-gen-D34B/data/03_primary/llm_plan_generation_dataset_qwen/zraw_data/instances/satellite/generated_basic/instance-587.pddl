(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph0 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
	star3 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(pointing satellite2 groundstation0)
	(pointing satellite3 groundstation0)
	(have_image star1 spectrograph0)
	(have_image phenomenon2 spectrograph0)
	(have_image star3 spectrograph0)
))

)
