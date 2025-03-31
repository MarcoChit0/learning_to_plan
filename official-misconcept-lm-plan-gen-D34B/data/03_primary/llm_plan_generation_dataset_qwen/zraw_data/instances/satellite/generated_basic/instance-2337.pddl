(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image2 - mode
	thermograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation4 - direction
	star1 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite0 star6)
	(have_image planet5 image2)
	(have_image star6 thermograph1)
	(have_image phenomenon7 infrared0)
	(have_image star8 thermograph1)
	(have_image phenomenon9 thermograph1)
))

)
