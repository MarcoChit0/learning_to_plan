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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	image0 - mode
	groundstation1 - direction
	groundstation5 - direction
	star0 - direction
	groundstation4 - direction
	star3 - direction
	groundstation2 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image0)
	(calibration_target instrument10 star0)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation2)
	(calibration_target instrument13 groundstation5)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation5)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument15 image0)
	(calibration_target instrument15 star0)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation5)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation5)
	(calibration_target instrument17 star0)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation5)
	(calibration_target instrument19 groundstation4)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation4)
	(calibration_target instrument20 star0)
	(supports instrument21 image0)
	(calibration_target instrument21 groundstation2)
	(calibration_target instrument21 star3)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation5)
)
(:goal (and
	(pointing satellite3 groundstation4)
	(have_image planet6 image0)
))

)
