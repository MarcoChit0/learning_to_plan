(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	image1 - mode
	infrared2 - mode
	spectrograph0 - mode
	star1 - direction
	star3 - direction
	groundstation4 - direction
	groundstation5 - direction
	groundstation2 - direction
	star0 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
	phenomenon9 - direction
	star10 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(have_image star6 infrared2)
	(have_image planet7 spectrograph0)
	(have_image star8 image1)
	(have_image phenomenon9 infrared2)
	(have_image star10 spectrograph0)
))

)
