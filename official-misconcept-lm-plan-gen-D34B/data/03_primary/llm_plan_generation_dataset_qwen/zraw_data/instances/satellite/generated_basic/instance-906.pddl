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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image1 - mode
	infrared0 - mode
	star3 - direction
	star0 - direction
	star2 - direction
	star1 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument10 infrared0)
	(supports instrument10 image1)
	(calibration_target instrument10 star1)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star0)
	(supports instrument12 image1)
	(calibration_target instrument12 star1)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument13 image1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(supports instrument14 image1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared0)
	(supports instrument15 image1)
	(calibration_target instrument15 star2)
	(supports instrument16 image1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star2)
	(supports instrument17 image1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star2)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 planet4)
	(pointing satellite2 phenomenon5)
	(pointing satellite3 star0)
	(have_image planet4 image1)
	(have_image phenomenon5 infrared0)
))

)
