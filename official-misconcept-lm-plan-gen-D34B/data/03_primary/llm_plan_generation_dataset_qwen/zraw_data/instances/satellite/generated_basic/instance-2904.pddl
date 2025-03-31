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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	infrared2 - mode
	infrared0 - mode
	image1 - mode
	infrared3 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 infrared3)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared3)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared2)
	(supports instrument5 infrared3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 infrared2)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument9 infrared3)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared3)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 infrared2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 infrared2)
	(supports instrument12 image1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 infrared0)
	(supports instrument13 infrared2)
	(calibration_target instrument13 star0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument14 image1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star0)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared2)
	(supports instrument16 infrared3)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image planet2 infrared0)
	(have_image star3 image1)
))

)
