(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	image2 - mode
	thermograph0 - mode
	spectrograph1 - mode
	star2 - direction
	groundstation3 - direction
	groundstation0 - direction
	star1 - direction
	star5 - direction
	star4 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star5)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 star4)
	(calibration_target instrument2 star5)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image planet6 image2)
	(have_image planet7 image2)
	(have_image planet8 thermograph0)
))

)
