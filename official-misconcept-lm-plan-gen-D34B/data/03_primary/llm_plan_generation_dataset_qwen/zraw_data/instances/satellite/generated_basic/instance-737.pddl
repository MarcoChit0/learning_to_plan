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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	thermograph0 - mode
	thermograph2 - mode
	star1 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	star2 - direction
	star3 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star5)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument7 image1)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star4)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star5)
	(supports instrument11 thermograph2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 image1)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite2 planet7)
	(pointing satellite3 star6)
	(have_image star6 image1)
	(have_image planet7 thermograph0)
	(have_image phenomenon8 image1)
	(have_image phenomenon9 thermograph2)
))

)
