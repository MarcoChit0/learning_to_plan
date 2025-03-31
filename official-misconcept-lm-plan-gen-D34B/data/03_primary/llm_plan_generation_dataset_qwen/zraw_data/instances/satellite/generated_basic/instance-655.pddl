(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph0 - mode
	infrared1 - mode
	groundstation5 - direction
	star1 - direction
	star2 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
)
(:goal (and
	(have_image phenomenon6 infrared1)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 spectrograph0)
))

)
