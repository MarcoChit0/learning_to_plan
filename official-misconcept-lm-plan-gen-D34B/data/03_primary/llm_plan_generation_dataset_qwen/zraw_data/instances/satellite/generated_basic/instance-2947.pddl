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
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	spectrograph3 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	image0 - mode
	star1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph2)
	(supports instrument5 image0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 spectrograph3)
	(supports instrument7 image0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star3)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite2 star1)
	(pointing satellite4 star3)
	(have_image star4 spectrograph3)
	(have_image planet5 image0)
))

)
