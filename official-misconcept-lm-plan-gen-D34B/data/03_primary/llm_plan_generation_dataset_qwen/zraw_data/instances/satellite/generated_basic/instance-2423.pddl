(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	thermograph3 - mode
	infrared2 - mode
	infrared0 - mode
	image1 - mode
	star0 - direction
	star1 - direction
	star3 - direction
	groundstation4 - direction
	groundstation2 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph3)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
	(supports instrument6 image1)
	(supports instrument6 thermograph3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
)
(:goal (and
	(pointing satellite3 groundstation4)
	(have_image planet5 image1)
	(have_image star6 image1)
))

)
