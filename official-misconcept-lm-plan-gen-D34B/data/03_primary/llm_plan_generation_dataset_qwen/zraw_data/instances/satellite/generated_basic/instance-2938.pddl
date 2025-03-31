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
	thermograph2 - mode
	infrared1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	star1 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star4)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image planet5 thermograph2)
	(have_image planet6 thermograph2)
	(have_image phenomenon7 thermograph2)
	(have_image star8 infrared0)
))

)
