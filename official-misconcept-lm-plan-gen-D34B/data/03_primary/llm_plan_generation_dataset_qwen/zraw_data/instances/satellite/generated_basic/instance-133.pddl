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
	thermograph0 - mode
	thermograph3 - mode
	image1 - mode
	infrared2 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph3)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 star3)
	(have_image phenomenon2 image1)
	(have_image star3 infrared2)
	(have_image phenomenon4 image1)
	(have_image star5 thermograph3)
	(have_image phenomenon6 thermograph3)
))

)
