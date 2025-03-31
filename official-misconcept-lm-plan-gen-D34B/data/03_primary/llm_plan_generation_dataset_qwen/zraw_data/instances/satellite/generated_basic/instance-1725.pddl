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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	thermograph0 - mode
	thermograph1 - mode
	star3 - direction
	groundstation4 - direction
	star1 - direction
	groundstation0 - direction
	groundstation2 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(have_image planet5 thermograph1)
))

)
