(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared1 - mode
	infrared2 - mode
	infrared3 - mode
	spectrograph0 - mode
	star1 - direction
	star3 - direction
	groundstation4 - direction
	star2 - direction
	groundstation0 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared3)
	(supports instrument2 infrared1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(have_image star5 infrared2)
	(have_image star6 infrared2)
	(have_image phenomenon7 infrared2)
	(have_image star8 spectrograph0)
))

)
