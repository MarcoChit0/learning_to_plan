(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star1 - direction
	star3 - direction
	star4 - direction
	star2 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
)
(:goal (and
	(have_image star5 spectrograph1)
	(have_image phenomenon6 spectrograph1)
	(have_image planet7 thermograph0)
	(have_image star8 thermograph0)
))

)
