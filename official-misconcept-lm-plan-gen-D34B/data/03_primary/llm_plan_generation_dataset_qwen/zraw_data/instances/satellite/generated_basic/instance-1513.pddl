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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	image1 - mode
	infrared0 - mode
	groundstation3 - direction
	star0 - direction
	star1 - direction
	groundstation5 - direction
	groundstation2 - direction
	groundstation4 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation3)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument9 image1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 star0)
	(supports instrument13 image1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation2)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation2)
	(calibration_target instrument14 groundstation5)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation3)
	(calibration_target instrument16 star1)
	(supports instrument17 image1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star1)
	(calibration_target instrument17 star0)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation5)
	(calibration_target instrument18 star1)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
	(supports instrument19 image1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation2)
	(supports instrument20 image1)
	(supports instrument20 infrared0)
	(calibration_target instrument20 groundstation4)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation4)
)
(:goal (and
	(pointing satellite3 phenomenon6)
	(pointing satellite5 star1)
	(have_image phenomenon6 image1)
	(have_image planet7 infrared0)
	(have_image phenomenon8 image1)
))

)
