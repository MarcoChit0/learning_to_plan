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
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	thermograph1 - mode
	image0 - mode
	image3 - mode
	image2 - mode
	groundstation2 - direction
	star0 - direction
	star1 - direction
	star3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image0)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image3)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 image0)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 image2)
	(supports instrument8 image3)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(supports instrument9 image3)
	(calibration_target instrument9 star3)
	(supports instrument10 image3)
	(supports instrument10 image0)
	(supports instrument10 image2)
	(calibration_target instrument10 star3)
	(supports instrument11 image3)
	(supports instrument11 image2)
	(calibration_target instrument11 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument12 image2)
	(supports instrument12 image0)
	(calibration_target instrument12 star3)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image3)
	(calibration_target instrument14 star0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument15 image3)
	(supports instrument15 thermograph1)
	(supports instrument15 image2)
	(calibration_target instrument15 star1)
	(supports instrument16 thermograph1)
	(supports instrument16 image2)
	(supports instrument16 image0)
	(calibration_target instrument16 star3)
	(supports instrument17 thermograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 star3)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite2 star1)
	(pointing satellite4 star0)
	(have_image phenomenon4 image3)
	(have_image planet5 image2)
))

)
