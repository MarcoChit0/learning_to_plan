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
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	spectrograph1 - mode
	thermograph3 - mode
	image2 - mode
	spectrograph0 - mode
	star0 - direction
	planet1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph1)
	(supports instrument8 thermograph3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
)
(:goal (and
	(have_image planet1 spectrograph0)
	(have_image planet2 image2)
	(have_image planet3 thermograph3)
))

)
