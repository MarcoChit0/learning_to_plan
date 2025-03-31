(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	spectrograph0 - mode
	infrared1 - mode
	infrared2 - mode
	star1 - direction
	groundstation0 - direction
	star4 - direction
	groundstation2 - direction
	groundstation5 - direction
	star3 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation5)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(have_image phenomenon6 infrared2)
	(have_image phenomenon7 infrared2)
	(have_image phenomenon8 infrared1)
	(have_image planet9 infrared1)
	(have_image planet10 spectrograph0)
))

)
