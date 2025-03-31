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
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	image0 - mode
	groundstation4 - direction
	star0 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation1 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument10 image0)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation3)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation2)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation2)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation1)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite4 planet7)
	(have_image star5 image0)
	(have_image phenomenon6 image0)
	(have_image planet7 image0)
))

)
