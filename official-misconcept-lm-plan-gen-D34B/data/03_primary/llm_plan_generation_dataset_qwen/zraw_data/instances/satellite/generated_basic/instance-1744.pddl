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
	thermograph2 - mode
	thermograph1 - mode
	infrared0 - mode
	groundstation4 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
	star1 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
	star10 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 planet6)
	(have_image planet6 thermograph1)
	(have_image star7 infrared0)
	(have_image phenomenon8 thermograph2)
	(have_image star9 thermograph2)
	(have_image star10 thermograph1)
))

)
