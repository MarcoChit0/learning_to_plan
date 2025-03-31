(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	spectrograph0 - mode
	spectrograph1 - mode
	groundstation5 - direction
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	star3 - direction
	groundstation4 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation4)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
)
(:goal (and
	(have_image star6 spectrograph1)
	(have_image star7 spectrograph0)
	(have_image planet8 spectrograph0)
))

)
