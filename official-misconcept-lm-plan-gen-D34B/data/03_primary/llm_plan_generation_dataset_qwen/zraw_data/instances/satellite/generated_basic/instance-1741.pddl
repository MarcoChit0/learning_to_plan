(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	infrared1 - mode
	thermograph0 - mode
	thermograph2 - mode
	star0 - direction
	groundstation4 - direction
	star5 - direction
	groundstation3 - direction
	star1 - direction
	star2 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon6 infrared1)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph2)
	(have_image star9 infrared1)
	(have_image phenomenon10 thermograph2)
))

)
