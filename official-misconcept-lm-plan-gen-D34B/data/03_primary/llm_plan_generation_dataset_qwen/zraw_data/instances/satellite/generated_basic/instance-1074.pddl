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
	satellite2 - satellite
	instrument5 - instrument
	thermograph1 - mode
	thermograph2 - mode
	image0 - mode
	star3 - direction
	groundstation5 - direction
	star0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star1 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph2)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(have_image planet6 thermograph1)
	(have_image planet7 image0)
	(have_image star8 thermograph2)
))

)
