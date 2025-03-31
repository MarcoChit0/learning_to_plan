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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	image0 - mode
	star3 - direction
	groundstation2 - direction
	groundstation5 - direction
	groundstation0 - direction
	groundstation4 - direction
	groundstation1 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 groundstation4)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(have_image planet6 image0)
	(have_image planet7 image0)
	(have_image planet8 image0)
))

)
