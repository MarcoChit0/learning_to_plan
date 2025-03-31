; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 329379

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child446 child465 - child
    bread353 bread226 - bread-portion
    content271 content176 - content-portion
    tray390 tray201 tray22 - tray
    table68 table427 table202 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at tray201 kitchen)
     (at tray22 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_bread bread226)
     (at_kitchen_content content271)
     (at_kitchen_content content176)
     (no_gluten_bread bread353)
     (no_gluten_content content176)
     (allergic_gluten child465)
     (not_allergic_gluten child446)
     (waiting child446 table202)
     (waiting child465 table427)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child446)
     (served child465)
    )
  )
)
