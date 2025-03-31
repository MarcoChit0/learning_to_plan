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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared0 - mode
	star5 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star2)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite3 star5)
	(have_image planet6 infrared0)
))

)
