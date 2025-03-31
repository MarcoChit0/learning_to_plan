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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	image0 - mode
	star0 - direction
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	groundstation1 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation4)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation4)
	(supports instrument16 image0)
	(calibration_target instrument16 star2)
	(supports instrument17 image0)
	(calibration_target instrument17 star2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation1)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet6)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 star7)
	(pointing satellite3 groundstation3)
	(pointing satellite4 star7)
	(have_image star5 image0)
	(have_image planet6 image0)
	(have_image star7 image0)
))

)
