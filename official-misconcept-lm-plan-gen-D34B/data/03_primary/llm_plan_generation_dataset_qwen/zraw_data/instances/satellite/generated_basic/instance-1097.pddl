(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image1 - mode
	thermograph0 - mode
	infrared2 - mode
	groundstation2 - direction
	star3 - direction
	star0 - direction
	groundstation1 - direction
	groundstation5 - direction
	star4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(supports instrument5 image1)
	(calibration_target instrument5 star4)
	(calibration_target instrument5 groundstation5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
)
(:goal (and
	(have_image phenomenon6 thermograph0)
	(have_image phenomenon7 infrared2)
	(have_image planet8 infrared2)
))

)
