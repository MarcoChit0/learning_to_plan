(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
)
(:goal (and
	(pointing satellite0 star8)
	(have_image planet5 spectrograph1)
	(have_image planet6 spectrograph1)
	(have_image phenomenon7 thermograph0)
	(have_image star8 thermograph0)
))

)
