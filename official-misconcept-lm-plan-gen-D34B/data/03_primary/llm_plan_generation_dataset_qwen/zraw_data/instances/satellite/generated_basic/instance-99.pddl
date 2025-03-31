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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star4 - direction
	groundstation5 - direction
	star1 - direction
	star2 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation5)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation5)
	(calibration_target instrument10 star1)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(calibration_target instrument11 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation5)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation0)
	(calibration_target instrument15 star2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument16 infrared0)
	(supports instrument16 image1)
	(calibration_target instrument16 star4)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 infrared0)
	(supports instrument17 image1)
	(calibration_target instrument17 star1)
	(calibration_target instrument17 groundstation5)
	(supports instrument18 image1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star2)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon8)
)
(:goal (and
	(pointing satellite4 groundstation0)
	(have_image phenomenon6 image1)
	(have_image star7 image1)
	(have_image phenomenon8 image1)
	(have_image planet9 image1)
))

)
