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
	infrared1 - mode
	image2 - mode
	thermograph0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	groundstation4 - direction
	groundstation3 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet10)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 image2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star5)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image phenomenon6 thermograph0)
	(have_image star7 image2)
	(have_image planet8 infrared1)
	(have_image phenomenon9 image2)
	(have_image planet10 thermograph0)
))

)
