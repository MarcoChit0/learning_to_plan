(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	satellite5 - satellite
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	instrument25 - instrument
	image0 - mode
	star3 - direction
	star2 - direction
	star4 - direction
	groundstation1 - direction
	groundstation0 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(calibration_target instrument6 star4)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image0)
	(calibration_target instrument10 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument11 image0)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(calibration_target instrument12 star4)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image0)
	(calibration_target instrument15 star2)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation0)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 image0)
	(calibration_target instrument19 star2)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation1)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument21 image0)
	(calibration_target instrument21 star4)
	(supports instrument22 image0)
	(calibration_target instrument22 star2)
	(supports instrument23 image0)
	(calibration_target instrument23 star4)
	(supports instrument24 image0)
	(calibration_target instrument24 groundstation1)
	(supports instrument25 image0)
	(calibration_target instrument25 groundstation0)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon5)
)
(:goal (and
	(pointing satellite2 groundstation1)
	(pointing satellite3 phenomenon5)
	(pointing satellite5 star2)
	(have_image phenomenon5 image0)
	(have_image planet6 image0)
))

)
