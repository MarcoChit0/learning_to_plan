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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph2 - mode
	image1 - mode
	thermograph0 - mode
	groundstation2 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	star1 - direction
	groundstation3 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star1)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 star4)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 image1)
	(supports instrument7 thermograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(calibration_target instrument9 star4)
	(supports instrument10 thermograph2)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star1)
	(calibration_target instrument11 star4)
	(supports instrument12 image1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star5)
	(calibration_target instrument12 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph2)
	(supports instrument14 image1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star5)
	(supports instrument15 image1)
	(calibration_target instrument15 star1)
	(calibration_target instrument15 star5)
	(supports instrument16 thermograph2)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation3)
	(calibration_target instrument16 star1)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet9)
)
(:goal (and
	(pointing satellite3 star7)
	(have_image star6 thermograph2)
	(have_image star7 image1)
	(have_image phenomenon8 image1)
	(have_image planet9 image1)
))

)
