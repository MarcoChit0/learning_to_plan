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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
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
	star3 - direction
	groundstation2 - direction
	groundstation0 - direction
	star5 - direction
	star1 - direction
	groundstation4 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 star5)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation4)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument11 image0)
	(calibration_target instrument11 star5)
	(calibration_target instrument11 star3)
	(supports instrument12 image0)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star5)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation4)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 image0)
	(calibration_target instrument16 star5)
	(supports instrument17 image0)
	(calibration_target instrument17 star5)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation4)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation4)
	(calibration_target instrument18 star1)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite3 groundstation4)
	(have_image planet6 image0)
))

)
