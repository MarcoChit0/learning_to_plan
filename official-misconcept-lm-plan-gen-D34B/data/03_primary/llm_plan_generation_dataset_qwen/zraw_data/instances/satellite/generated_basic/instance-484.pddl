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
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	image2 - mode
	thermograph1 - mode
	image0 - mode
	infrared3 - mode
	star1 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation2 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 infrared3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 image2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared3)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image0)
	(supports instrument11 infrared3)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument12 infrared3)
	(supports instrument12 image2)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared3)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 image2)
	(supports instrument15 infrared3)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 image2)
	(calibration_target instrument16 groundstation4)
	(supports instrument17 image0)
	(supports instrument17 infrared3)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
)
(:goal (and
	(pointing satellite2 planet6)
	(have_image phenomenon5 infrared3)
	(have_image planet6 thermograph1)
	(have_image star7 thermograph1)
))

)
