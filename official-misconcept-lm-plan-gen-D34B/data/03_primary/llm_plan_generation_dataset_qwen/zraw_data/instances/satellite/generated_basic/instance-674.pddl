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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared1 - mode
	infrared0 - mode
	star2 - direction
	groundstation4 - direction
	groundstation5 - direction
	groundstation0 - direction
	star1 - direction
	star3 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation5)
	(have_image star6 infrared1)
	(have_image star7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
