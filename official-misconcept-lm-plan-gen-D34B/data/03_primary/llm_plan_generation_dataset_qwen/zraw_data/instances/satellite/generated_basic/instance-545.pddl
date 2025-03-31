(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	spectrograph3 - mode
	image2 - mode
	star1 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation4 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image planet5 image2)
	(have_image planet6 spectrograph3)
	(have_image star7 image2)
	(have_image star8 image2)
))

)
