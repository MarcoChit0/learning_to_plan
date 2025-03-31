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
	thermograph1 - mode
	spectrograph0 - mode
	star5 - direction
	star3 - direction
	groundstation0 - direction
	star2 - direction
	star1 - direction
	star4 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star9)
	(supports instrument2 spectrograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite2 star6)
	(have_image star6 thermograph1)
	(have_image planet7 thermograph1)
	(have_image phenomenon8 image2)
	(have_image star9 image2)
))

)
