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
	instrument5 - instrument
	thermograph1 - mode
	thermograph2 - mode
	thermograph3 - mode
	thermograph0 - mode
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	groundstation0 - direction
	star1 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
	star10 - direction
	planet11 - direction
	star12 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet11)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image planet5 thermograph1)
	(have_image planet6 thermograph1)
	(have_image star7 thermograph3)
	(have_image phenomenon8 thermograph2)
	(have_image planet9 thermograph2)
	(have_image star10 thermograph2)
	(have_image planet11 thermograph3)
	(have_image star12 thermograph1)
))

)
