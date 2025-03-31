(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	thermograph0 - mode
	groundstation4 - direction
	groundstation1 - direction
	star2 - direction
	star0 - direction
	groundstation3 - direction
	groundstation5 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 groundstation5)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(have_image star6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image planet8 thermograph0)
))

)
