(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
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
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	groundstation5 - direction
	groundstation4 - direction
	star0 - direction
	groundstation3 - direction
	star1 - direction
	star2 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation3)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation5)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(calibration_target instrument12 star1)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 image0)
	(calibration_target instrument15 star1)
	(calibration_target instrument15 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation3)
	(calibration_target instrument17 star0)
	(supports instrument18 image0)
	(calibration_target instrument18 star1)
	(supports instrument19 image0)
	(calibration_target instrument19 star2)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image phenomenon6 image0)
	(have_image planet7 image0)
	(have_image phenomenon8 image0)
))

)
