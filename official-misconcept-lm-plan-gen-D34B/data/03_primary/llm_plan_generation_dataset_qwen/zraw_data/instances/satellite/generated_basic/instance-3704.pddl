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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	infrared2 - mode
	thermograph0 - mode
	image1 - mode
	groundstation5 - direction
	star4 - direction
	groundstation1 - direction
	groundstation2 - direction
	star0 - direction
	groundstation3 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite1 star6)
	(pointing satellite2 groundstation1)
	(have_image star6 image1)
))

)
