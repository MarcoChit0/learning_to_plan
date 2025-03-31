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
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	image0 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation5 - direction
	star2 - direction
	star1 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image0)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument12 image0)
	(calibration_target instrument12 star1)
	(calibration_target instrument12 groundstation3)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star6)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation5)
	(calibration_target instrument13 star2)
	(supports instrument14 image0)
	(calibration_target instrument14 star1)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 image0)
	(calibration_target instrument15 star2)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 image0)
	(calibration_target instrument16 star1)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 groundstation0)
	(pointing satellite3 groundstation5)
	(have_image star6 image0)
))

)
