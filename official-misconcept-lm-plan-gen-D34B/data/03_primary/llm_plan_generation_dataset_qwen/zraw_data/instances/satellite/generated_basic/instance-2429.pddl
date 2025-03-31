(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	spectrograph0 - mode
	image2 - mode
	thermograph1 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
	planet3 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(have_image phenomenon2 thermograph1)
	(have_image planet3 spectrograph0)
	(have_image star4 spectrograph0)
	(have_image star5 spectrograph0)
))

)
