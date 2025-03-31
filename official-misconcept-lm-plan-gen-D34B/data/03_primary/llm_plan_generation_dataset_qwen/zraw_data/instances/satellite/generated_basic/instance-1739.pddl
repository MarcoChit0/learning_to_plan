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
	instrument5 - instrument
	image0 - mode
	infrared2 - mode
	thermograph1 - mode
	star0 - direction
	groundstation4 - direction
	star1 - direction
	groundstation5 - direction
	groundstation3 - direction
	star2 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
	phenomenon9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
)
(:goal (and
	(pointing satellite0 planet8)
	(pointing satellite1 star2)
	(have_image star6 thermograph1)
	(have_image planet7 image0)
	(have_image planet8 image0)
	(have_image phenomenon9 image0)
	(have_image planet10 infrared2)
))

)
