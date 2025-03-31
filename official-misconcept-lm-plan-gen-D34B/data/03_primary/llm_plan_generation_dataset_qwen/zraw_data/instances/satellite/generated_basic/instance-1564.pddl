(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	infrared0 - mode
	image1 - mode
	infrared2 - mode
	groundstation0 - direction
	phenomenon1 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image1)
	(supports instrument3 infrared2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument6 image1)
	(supports instrument6 infrared2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon1)
	(supports instrument12 image1)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 image1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared0)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared2)
	(supports instrument16 image1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite3 phenomenon1)
	(have_image phenomenon1 infrared0)
))

)
