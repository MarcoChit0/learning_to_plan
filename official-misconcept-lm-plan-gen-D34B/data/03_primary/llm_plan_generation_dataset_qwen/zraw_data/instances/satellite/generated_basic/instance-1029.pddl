(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	infrared1 - mode
	infrared2 - mode
	spectrograph0 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation2 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image planet4 infrared1)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 spectrograph0)
	(have_image star7 infrared2)
	(have_image phenomenon8 infrared2)
))

)
