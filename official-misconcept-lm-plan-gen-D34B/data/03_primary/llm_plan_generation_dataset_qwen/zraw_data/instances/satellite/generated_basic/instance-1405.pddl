(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	thermograph2 - mode
	infrared1 - mode
	image3 - mode
	star0 - direction
	groundstation2 - direction
	star1 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 image3)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(supports instrument2 image3)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument5 image3)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star1)
	(supports instrument6 image3)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image3)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(supports instrument9 image3)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared1)
	(supports instrument10 image3)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph2)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument12 image3)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star1)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument13 image0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph2)
	(supports instrument15 infrared1)
	(supports instrument15 image3)
	(calibration_target instrument15 groundstation2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument16 image0)
	(supports instrument16 infrared1)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 image3)
	(supports instrument17 image0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star1)
	(supports instrument18 image0)
	(supports instrument18 thermograph2)
	(supports instrument18 image3)
	(calibration_target instrument18 star1)
	(supports instrument19 image0)
	(supports instrument19 infrared1)
	(supports instrument19 thermograph2)
	(calibration_target instrument19 star1)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite2 planet6)
	(pointing satellite3 groundstation2)
	(have_image star3 image3)
	(have_image star4 image0)
	(have_image planet5 infrared1)
	(have_image planet6 image0)
	(have_image phenomenon7 image0)
))

)
