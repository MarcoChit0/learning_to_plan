(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation5 - direction
	star2 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation0 - direction
	star3 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation5)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet9)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 star2)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation5)
	(calibration_target instrument9 groundstation1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star3)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 groundstation4)
	(supports instrument13 thermograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image1)
	(calibration_target instrument14 star3)
	(calibration_target instrument14 groundstation0)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation5)
)
(:goal (and
	(have_image phenomenon6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image phenomenon8 image1)
	(have_image planet9 thermograph0)
))

)
