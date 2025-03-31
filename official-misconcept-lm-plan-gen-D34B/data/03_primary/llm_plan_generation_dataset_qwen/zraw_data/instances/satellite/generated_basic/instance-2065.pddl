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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	image0 - mode
	groundstation2 - direction
	star1 - direction
	star3 - direction
	groundstation0 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image0)
	(calibration_target instrument15 star1)
	(supports instrument16 image0)
	(calibration_target instrument16 star3)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation0)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star1)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image phenomenon4 image0)
))

)
