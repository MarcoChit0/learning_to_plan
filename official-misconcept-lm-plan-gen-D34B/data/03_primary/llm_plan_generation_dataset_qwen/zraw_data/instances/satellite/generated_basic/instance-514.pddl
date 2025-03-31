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
	instrument5 - instrument
	image1 - mode
	infrared0 - mode
	thermograph2 - mode
	infrared3 - mode
	groundstation2 - direction
	star3 - direction
	star4 - direction
	groundstation1 - direction
	groundstation0 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(supports instrument4 infrared0)
	(supports instrument4 infrared3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(pointing satellite1 star3)
	(have_image planet5 infrared3)
	(have_image phenomenon6 image1)
	(have_image phenomenon7 thermograph2)
	(have_image phenomenon8 image1)
))

)
