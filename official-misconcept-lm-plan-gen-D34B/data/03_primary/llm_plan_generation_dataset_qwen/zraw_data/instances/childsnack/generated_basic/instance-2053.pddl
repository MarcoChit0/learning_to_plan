; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 163317

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 child144 - child
    bread377 bread319 - bread-portion
    content157 content49 - content-portion
    tray244 tray198 tray452 - tray
    table461 table205 table75 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at tray198 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread319)
     (at_kitchen_content content157)
     (at_kitchen_content content49)
     (not_allergic_gluten child144)
     (not_allergic_gluten child103)
     (waiting child103 table461)
     (waiting child144 table75)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child103)
     (served child144)
    )
  )
)
