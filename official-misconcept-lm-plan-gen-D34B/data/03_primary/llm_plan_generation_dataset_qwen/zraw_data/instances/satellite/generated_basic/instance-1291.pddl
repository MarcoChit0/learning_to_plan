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
	thermograph0 - mode
	thermograph2 - mode
	thermograph1 - mode
	star3 - direction
	groundstation5 - direction
	star2 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation1 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 planet9)
	(have_image phenomenon6 thermograph0)
	(have_image phenomenon7 thermograph2)
	(have_image phenomenon8 thermograph2)
	(have_image planet9 thermograph0)
))

)
