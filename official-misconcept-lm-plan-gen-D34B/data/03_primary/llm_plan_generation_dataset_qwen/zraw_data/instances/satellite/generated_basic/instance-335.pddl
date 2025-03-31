(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	image3 - mode
	thermograph0 - mode
	thermograph1 - mode
	infrared2 - mode
	groundstation2 - direction
	star3 - direction
	star5 - direction
	star0 - direction
	star4 - direction
	groundstation1 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(supports instrument2 image3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star4)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 star3)
	(have_image planet6 thermograph1)
))

)
