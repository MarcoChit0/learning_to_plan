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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	infrared0 - mode
	star1 - direction
	star3 - direction
	groundstation0 - direction
	groundstation2 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(pointing satellite0 phenomenon5)
	(pointing satellite2 star4)
	(have_image star4 infrared0)
	(have_image phenomenon5 infrared0)
	(have_image planet6 infrared0)
))

)
