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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
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
	instrument21 - instrument
	instrument22 - instrument
	image0 - mode
	star1 - direction
	star3 - direction
	groundstation4 - direction
	star2 - direction
	groundstation0 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation4)
	(supports instrument17 image0)
	(calibration_target instrument17 star3)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet9)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation4)
	(supports instrument19 image0)
	(calibration_target instrument19 star2)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation0)
	(supports instrument21 image0)
	(calibration_target instrument21 star2)
	(supports instrument22 image0)
	(calibration_target instrument22 groundstation0)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet10)
)
(:goal (and
	(pointing satellite1 star3)
	(pointing satellite2 planet10)
	(pointing satellite4 groundstation4)
	(pointing satellite5 planet9)
	(have_image star5 image0)
	(have_image planet6 image0)
	(have_image star7 image0)
	(have_image star8 image0)
	(have_image planet9 image0)
	(have_image planet10 image0)
))

)
