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
	satellite3 - satellite
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	thermograph0 - mode
	groundstation2 - direction
	groundstation3 - direction
	star5 - direction
	groundstation1 - direction
	star4 - direction
	groundstation0 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star5)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star4)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(have_image star6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
