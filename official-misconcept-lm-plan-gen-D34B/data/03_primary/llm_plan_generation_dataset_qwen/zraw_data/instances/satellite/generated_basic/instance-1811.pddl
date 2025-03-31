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
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image0 - mode
	thermograph2 - mode
	infrared3 - mode
	infrared1 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared3)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared3)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument11 infrared1)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared1)
	(supports instrument12 infrared3)
	(supports instrument12 image0)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star1)
	(supports instrument14 infrared1)
	(supports instrument14 infrared3)
	(supports instrument14 image0)
	(calibration_target instrument14 star2)
	(supports instrument15 image0)
	(calibration_target instrument15 star0)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 planet4)
	(have_image phenomenon3 infrared3)
	(have_image planet4 infrared3)
))

)
