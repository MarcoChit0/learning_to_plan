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
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	thermograph2 - mode
	image0 - mode
	infrared1 - mode
	star5 - direction
	star1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
	star4 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star3)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 star5)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star5)
	(supports instrument8 image0)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument9 image0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star5)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared1)
	(supports instrument11 thermograph2)
	(supports instrument11 image0)
	(calibration_target instrument11 star3)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph2)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star4)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image phenomenon6 image0)
	(have_image planet7 image0)
	(have_image phenomenon8 image0)
	(have_image star9 thermograph2)
))

)
