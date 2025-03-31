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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	star1 - direction
	groundstation4 - direction
	groundstation5 - direction
	star2 - direction
	groundstation0 - direction
	groundstation3 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image star6 image0)
))

)
