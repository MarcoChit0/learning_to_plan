(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	thermograph2 - mode
	infrared0 - mode
	star5 - direction
	star4 - direction
	groundstation3 - direction
	star1 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph2)
	(supports instrument7 image1)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star4)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 image1)
	(supports instrument12 thermograph2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon6 thermograph2)
	(have_image star7 image1)
	(have_image phenomenon8 thermograph2)
	(have_image phenomenon9 infrared0)
))

)
