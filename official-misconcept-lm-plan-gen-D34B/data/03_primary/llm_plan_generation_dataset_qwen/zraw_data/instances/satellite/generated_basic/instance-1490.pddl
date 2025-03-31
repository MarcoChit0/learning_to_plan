(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	thermograph1 - mode
	star0 - direction
	star2 - direction
	star3 - direction
	groundstation4 - direction
	groundstation5 - direction
	groundstation1 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star3)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation1)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star3)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph1)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation1)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation5)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation1)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 groundstation4)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 thermograph1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation1)
	(calibration_target instrument19 groundstation5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon7)
)
(:goal (and
	(pointing satellite0 groundstation5)
	(pointing satellite3 phenomenon6)
	(pointing satellite4 star8)
	(have_image phenomenon6 image0)
	(have_image phenomenon7 image0)
	(have_image star8 thermograph1)
))

)
