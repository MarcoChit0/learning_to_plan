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
	satellite2 - satellite
	instrument5 - instrument
	infrared2 - mode
	infrared1 - mode
	spectrograph3 - mode
	spectrograph0 - mode
	groundstation1 - direction
	groundstation0 - direction
	groundstation5 - direction
	star3 - direction
	groundstation4 - direction
	star2 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(have_image star6 infrared1)
))

)
