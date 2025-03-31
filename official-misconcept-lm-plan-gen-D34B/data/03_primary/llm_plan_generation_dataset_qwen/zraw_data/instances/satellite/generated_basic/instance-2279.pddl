(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image1 - mode
	spectrograph0 - mode
	infrared3 - mode
	thermograph2 - mode
	star0 - direction
	planet1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph2)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph2)
	(supports instrument12 image1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared3)
	(supports instrument13 spectrograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 planet1)
	(pointing satellite2 phenomenon3)
	(have_image planet1 infrared3)
	(have_image star2 infrared3)
	(have_image phenomenon3 image1)
))

)
