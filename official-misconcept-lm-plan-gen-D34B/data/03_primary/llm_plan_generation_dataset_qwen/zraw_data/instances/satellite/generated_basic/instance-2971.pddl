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
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite4 - satellite
	instrument18 - instrument
	infrared1 - mode
	infrared3 - mode
	image0 - mode
	infrared2 - mode
	star3 - direction
	groundstation1 - direction
	star2 - direction
	star0 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared3)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared2)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared3)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument10 infrared2)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared2)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument12 infrared3)
	(supports instrument12 image0)
	(supports instrument12 infrared2)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared2)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared3)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image0)
	(supports instrument15 infrared1)
	(supports instrument15 infrared2)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared1)
	(supports instrument16 image0)
	(supports instrument16 infrared3)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument18 infrared3)
	(calibration_target instrument18 star0)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 star5)
	(pointing satellite4 star5)
	(have_image planet4 infrared3)
	(have_image star5 infrared2)
))

)
