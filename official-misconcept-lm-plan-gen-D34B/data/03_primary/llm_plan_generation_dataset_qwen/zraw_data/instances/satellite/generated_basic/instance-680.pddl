(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	infrared1 - mode
	spectrograph0 - mode
	groundstation0 - direction
	groundstation5 - direction
	star3 - direction
	star1 - direction
	groundstation2 - direction
	star4 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite1 planet7)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 infrared1)
	(have_image star8 spectrograph0)
))

)
