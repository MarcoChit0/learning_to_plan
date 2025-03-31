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
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite3 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	image0 - mode
	infrared2 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation2 - direction
	star0 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 infrared2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared2)
	(supports instrument5 infrared1)
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
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument11 infrared2)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared2)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image0)
	(supports instrument14 infrared1)
	(supports instrument14 infrared2)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared2)
	(calibration_target instrument15 groundstation1)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument16 infrared2)
	(supports instrument16 infrared1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 image0)
	(supports instrument18 infrared2)
	(calibration_target instrument18 star0)
	(supports instrument19 infrared2)
	(calibration_target instrument19 star0)
	(supports instrument20 infrared1)
	(calibration_target instrument20 star0)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(on_board instrument20 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 star0)
	(pointing satellite3 planet3)
	(have_image planet3 infrared2)
))

)
