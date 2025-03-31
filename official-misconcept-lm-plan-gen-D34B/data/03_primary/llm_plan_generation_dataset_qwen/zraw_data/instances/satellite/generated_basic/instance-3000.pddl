(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	star1 - direction
	groundstation5 - direction
	star3 - direction
	groundstation2 - direction
	groundstation4 - direction
	groundstation0 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 planet6)
	(have_image planet6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
