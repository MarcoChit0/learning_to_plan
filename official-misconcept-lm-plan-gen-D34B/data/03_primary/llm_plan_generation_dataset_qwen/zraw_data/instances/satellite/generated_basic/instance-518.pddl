(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	infrared3 - mode
	infrared1 - mode
	thermograph2 - mode
	image0 - mode
	groundstation0 - direction
	groundstation2 - direction
	star4 - direction
	star3 - direction
	groundstation1 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(have_image planet5 infrared1)
	(have_image planet6 thermograph2)
	(have_image phenomenon7 infrared1)
	(have_image star8 thermograph2)
))

)
