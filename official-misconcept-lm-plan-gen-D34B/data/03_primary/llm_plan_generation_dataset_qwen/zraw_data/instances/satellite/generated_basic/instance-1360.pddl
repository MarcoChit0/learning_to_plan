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
	instrument6 - instrument
	image3 - mode
	infrared0 - mode
	infrared2 - mode
	image1 - mode
	groundstation0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image3)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared2)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite1 planet1)
	(have_image planet1 infrared0)
))

)
