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
	instrument16 - instrument
	satellite3 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	satellite4 - satellite
	instrument22 - instrument
	instrument23 - instrument
	satellite5 - satellite
	instrument24 - instrument
	infrared0 - mode
	infrared1 - mode
	star1 - direction
	groundstation0 - direction
	phenomenon2 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument11 infrared0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star1)
	(supports instrument14 infrared1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star1)
	(supports instrument15 infrared1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star1)
	(supports instrument16 infrared1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation0)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(on_board instrument16 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(supports instrument19 infrared1)
	(calibration_target instrument19 star1)
	(supports instrument20 infrared1)
	(calibration_target instrument20 star1)
	(supports instrument21 infrared1)
	(supports instrument21 infrared0)
	(calibration_target instrument21 groundstation0)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(on_board instrument20 satellite3)
	(on_board instrument21 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument22 infrared1)
	(calibration_target instrument22 star1)
	(supports instrument23 infrared1)
	(supports instrument23 infrared0)
	(calibration_target instrument23 groundstation0)
	(on_board instrument22 satellite4)
	(on_board instrument23 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon2)
	(supports instrument24 infrared0)
	(supports instrument24 infrared1)
	(calibration_target instrument24 groundstation0)
	(on_board instrument24 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(pointing satellite2 star1)
	(have_image phenomenon2 infrared0)
	(have_image star3 infrared1)
	(have_image star4 infrared1)
))

)
