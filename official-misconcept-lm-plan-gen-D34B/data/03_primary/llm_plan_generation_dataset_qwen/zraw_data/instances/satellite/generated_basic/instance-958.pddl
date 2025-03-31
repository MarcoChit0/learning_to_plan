(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	image0 - mode
	spectrograph3 - mode
	thermograph2 - mode
	infrared1 - mode
	star1 - direction
	groundstation3 - direction
	star0 - direction
	star2 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph3)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(supports instrument8 spectrograph3)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument9 image0)
	(supports instrument9 spectrograph3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(pointing satellite2 star6)
	(have_image phenomenon4 infrared1)
	(have_image phenomenon5 image0)
	(have_image star6 spectrograph3)
))

)
