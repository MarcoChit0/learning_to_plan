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
	image1 - mode
	thermograph0 - mode
	infrared2 - mode
	groundstation0 - direction
	phenomenon1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(have_image phenomenon1 image1)
	(have_image planet2 infrared2)
))

)
