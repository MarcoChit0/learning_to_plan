(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	spectrograph2 - mode
	spectrograph1 - mode
	thermograph0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	star1 - direction
	groundstation4 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(have_image star5 spectrograph1)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph0)
	(have_image star8 thermograph0)
	(have_image phenomenon9 spectrograph2)
))

)
