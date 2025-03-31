(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	spectrograph3 - mode
	image0 - mode
	infrared2 - mode
	groundstation2 - direction
	star1 - direction
	star4 - direction
	star0 - direction
	groundstation3 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph3)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(supports instrument4 infrared2)
	(supports instrument4 image0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument5 spectrograph3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star4)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite2 star7)
	(have_image planet5 image1)
	(have_image star6 image0)
	(have_image star7 image1)
))

)
