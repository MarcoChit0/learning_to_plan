; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 787223

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 child405 child25 - child
    bread133 bread451 bread206 - bread-portion
    content222 content104 content324 - content-portion
    tray71 tray139 tray417 - tray
    table390 table33 table372 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray71 kitchen)
     (at tray139 kitchen)
     (at tray417 kitchen)
     (at_kitchen_bread bread133)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread206)
     (at_kitchen_content content222)
     (at_kitchen_content content104)
     (at_kitchen_content content324)
     (not_allergic_gluten child25)
     (not_allergic_gluten child405)
     (not_allergic_gluten child90)
     (waiting child90 table33)
     (waiting child405 table33)
     (waiting child25 table372)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child90)
     (served child405)
     (served child25)
    )
  )
)
