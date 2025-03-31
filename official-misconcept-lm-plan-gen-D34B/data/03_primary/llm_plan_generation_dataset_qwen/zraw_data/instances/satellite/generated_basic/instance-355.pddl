(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite1 star3)
	(pointing satellite2 planet5)
	(have_image phenomenon4 image0)
	(have_image planet5 infrared1)
	(have_image planet6 image0)
))

)
