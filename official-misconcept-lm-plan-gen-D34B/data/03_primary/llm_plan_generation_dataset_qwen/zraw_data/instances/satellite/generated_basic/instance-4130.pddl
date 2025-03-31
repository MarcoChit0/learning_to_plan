(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
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
	instrument16 - instrument
	thermograph1 - mode
	image0 - mode
	star5 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star5)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 star5)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation1)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument11 thermograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 image0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation4)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation4)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph1)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation3)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation0)
	(calibration_target instrument16 groundstation2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(on_board instrument16 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image planet6 image0)
))

)
