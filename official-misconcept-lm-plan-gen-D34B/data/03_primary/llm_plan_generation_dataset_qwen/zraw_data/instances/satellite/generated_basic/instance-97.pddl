(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	image0 - mode
	thermograph1 - mode
	groundstation2 - direction
	groundstation5 - direction
	groundstation1 - direction
	star3 - direction
	star4 - direction
	star0 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 star3)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation5)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 image0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 star4)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument12 thermograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation5)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation1)
	(calibration_target instrument13 groundstation5)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star4)
	(calibration_target instrument14 star3)
	(supports instrument15 image0)
	(calibration_target instrument15 star0)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(calibration_target instrument16 star4)
	(supports instrument17 thermograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 star0)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon7)
)
(:goal (and
	(pointing satellite2 groundstation5)
	(pointing satellite4 star6)
	(have_image star6 image0)
	(have_image phenomenon7 image0)
	(have_image star8 image0)
	(have_image star9 image0)
))

)
