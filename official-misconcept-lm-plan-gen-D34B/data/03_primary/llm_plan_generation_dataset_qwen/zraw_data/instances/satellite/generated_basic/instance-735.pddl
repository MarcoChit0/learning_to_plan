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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	thermograph0 - mode
	image2 - mode
	thermograph1 - mode
	groundstation5 - direction
	groundstation0 - direction
	groundstation4 - direction
	groundstation3 - direction
	star1 - direction
	groundstation2 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image2)
	(supports instrument7 thermograph0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation3)
	(calibration_target instrument7 star1)
	(supports instrument8 image2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 star1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 thermograph1)
	(supports instrument13 image2)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 image2)
	(calibration_target instrument14 groundstation2)
	(calibration_target instrument14 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image planet6 thermograph1)
	(have_image planet7 thermograph1)
	(have_image phenomenon8 thermograph0)
	(have_image phenomenon9 thermograph1)
))

)
