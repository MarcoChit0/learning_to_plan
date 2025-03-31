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
	instrument8 - instrument
	infrared0 - mode
	infrared1 - mode
	infrared2 - mode
	infrared3 - mode
	star3 - direction
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared3)
	(supports instrument1 infrared2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared2)
	(supports instrument7 infrared3)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(have_image planet4 infrared2)
))

)
