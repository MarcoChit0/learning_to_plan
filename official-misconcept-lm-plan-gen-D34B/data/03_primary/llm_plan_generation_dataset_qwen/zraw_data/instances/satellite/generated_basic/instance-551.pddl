(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	infrared2 - mode
	spectrograph1 - mode
	image0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(have_image star2 spectrograph1)
	(have_image star3 infrared2)
	(have_image star4 image0)
	(have_image planet5 infrared2)
	(have_image planet6 image0)
))

)
