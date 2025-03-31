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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared1 - mode
	spectrograph0 - mode
	image2 - mode
	groundstation4 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 image2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 image2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
	(supports instrument7 spectrograph0)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 spectrograph0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation5)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation3)
	(calibration_target instrument13 groundstation5)
	(supports instrument14 infrared1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 infrared1)
	(supports instrument15 image2)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 spectrograph0)
	(supports instrument16 image2)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation1)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 image2)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation5)
	(calibration_target instrument17 groundstation0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 image2)
	(have_image planet8 image2)
	(have_image phenomenon9 spectrograph0)
))

)
