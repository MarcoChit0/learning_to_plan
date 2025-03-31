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
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	image0 - mode
	thermograph1 - mode
	groundstation0 - direction
	groundstation3 - direction
	star5 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation4 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star5)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star9)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 groundstation4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 thermograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 star5)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star5)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation4)
	(supports instrument13 thermograph1)
	(supports instrument13 image0)
	(calibration_target instrument13 star5)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 image0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation4)
	(calibration_target instrument17 groundstation2)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(have_image phenomenon6 thermograph1)
	(have_image star7 image0)
	(have_image planet8 image0)
	(have_image star9 image0)
))

)
