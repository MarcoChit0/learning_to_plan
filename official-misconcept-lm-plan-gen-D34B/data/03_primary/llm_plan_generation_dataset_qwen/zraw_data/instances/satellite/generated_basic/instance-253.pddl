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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	image0 - mode
	image1 - mode
	star2 - direction
	star3 - direction
	groundstation4 - direction
	star1 - direction
	groundstation0 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 star3)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument11 image0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 image0)
	(supports instrument12 image1)
	(calibration_target instrument12 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet8)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation0)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation4)
)
(:goal (and
	(pointing satellite0 planet6)
	(have_image planet5 image0)
	(have_image planet6 image1)
	(have_image planet7 image1)
	(have_image planet8 image0)
))

)
