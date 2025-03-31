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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	infrared1 - mode
	image0 - mode
	infrared2 - mode
	groundstation1 - direction
	star0 - direction
	phenomenon2 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared2)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument1 infrared2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 infrared2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument7 image0)
	(supports instrument7 infrared2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(supports instrument10 image0)
	(supports instrument10 infrared1)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image0)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(supports instrument13 infrared1)
	(supports instrument13 infrared2)
	(calibration_target instrument13 star0)
	(supports instrument14 infrared1)
	(supports instrument14 image0)
	(supports instrument14 infrared2)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared2)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
	(supports instrument16 image0)
	(supports instrument16 infrared2)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared1)
	(supports instrument17 image0)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared2)
	(supports instrument18 infrared1)
	(supports instrument18 image0)
	(calibration_target instrument18 star0)
	(supports instrument19 infrared1)
	(supports instrument19 infrared2)
	(supports instrument19 image0)
	(calibration_target instrument19 star0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet3)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite3 planet3)
	(have_image phenomenon2 infrared1)
	(have_image planet3 image0)
	(have_image phenomenon4 infrared2)
))

)
