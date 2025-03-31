(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	image1 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image star3 image1)
	(have_image planet4 infrared0)
))

)
