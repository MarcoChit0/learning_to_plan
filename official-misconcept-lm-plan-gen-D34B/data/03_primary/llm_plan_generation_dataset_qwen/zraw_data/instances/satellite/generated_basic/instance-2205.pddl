(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph0 - mode
	infrared1 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
)
(:goal (and
	(have_image star4 spectrograph0)
	(have_image star5 spectrograph0)
	(have_image planet6 spectrograph0)
	(have_image star7 infrared1)
	(have_image star8 infrared1)
	(have_image planet9 infrared1)
))

)
