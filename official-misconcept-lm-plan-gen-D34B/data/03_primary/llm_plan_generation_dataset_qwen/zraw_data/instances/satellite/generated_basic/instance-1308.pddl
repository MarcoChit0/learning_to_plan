(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	infrared2 - mode
	spectrograph1 - mode
	image0 - mode
	star2 - direction
	star5 - direction
	groundstation3 - direction
	star1 - direction
	star4 - direction
	groundstation0 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star4)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite2 star2)
	(have_image star6 image0)
	(have_image planet7 spectrograph1)
	(have_image star8 image0)
	(have_image planet9 spectrograph1)
))

)
