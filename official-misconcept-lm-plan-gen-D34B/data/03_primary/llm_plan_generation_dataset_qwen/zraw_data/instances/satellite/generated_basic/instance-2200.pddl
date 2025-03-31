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
	thermograph0 - mode
	thermograph1 - mode
	star0 - direction
	star2 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
)
(:goal (and
	(pointing satellite1 star3)
	(have_image phenomenon4 thermograph1)
	(have_image phenomenon5 thermograph0)
	(have_image planet6 thermograph0)
	(have_image phenomenon7 thermograph1)
	(have_image planet8 thermograph1)
	(have_image phenomenon9 thermograph1)
))

)
