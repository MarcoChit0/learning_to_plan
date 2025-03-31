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
	satellite3 - satellite
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	image1 - mode
	infrared0 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
	(supports instrument10 infrared0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
	(supports instrument11 infrared0)
	(supports instrument11 image1)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared0)
	(supports instrument14 image1)
	(calibration_target instrument14 star1)
	(supports instrument15 image1)
	(calibration_target instrument15 star1)
	(supports instrument16 image1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star1)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star3)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(pointing satellite2 phenomenon4)
	(pointing satellite3 phenomenon2)
	(have_image phenomenon2 infrared0)
	(have_image star3 image1)
	(have_image phenomenon4 infrared0)
))

)
