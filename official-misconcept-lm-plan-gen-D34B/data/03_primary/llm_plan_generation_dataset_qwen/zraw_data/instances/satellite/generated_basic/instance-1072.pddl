(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	image0 - mode
	infrared1 - mode
	thermograph2 - mode
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	star2 - direction
	groundstation4 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument1 thermograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star5)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 groundstation4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 groundstation4)
	(have_image planet6 infrared1)
	(have_image planet7 image0)
	(have_image phenomenon8 thermograph2)
))

)
