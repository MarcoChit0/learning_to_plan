(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	spectrograph0 - mode
	star3 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation5 - direction
	groundstation4 - direction
	groundstation0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 groundstation2)
	(have_image planet6 spectrograph0)
))

)
