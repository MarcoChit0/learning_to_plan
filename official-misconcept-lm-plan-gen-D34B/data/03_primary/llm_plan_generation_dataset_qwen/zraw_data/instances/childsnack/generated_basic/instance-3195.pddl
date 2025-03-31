; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 974366

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 child160 - child
    bread91 bread143 - bread-portion
    content125 content454 - content-portion
    tray352 tray467 - tray
    table94 table191 table78 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread91)
     (at_kitchen_bread bread143)
     (at_kitchen_content content125)
     (at_kitchen_content content454)
     (no_gluten_bread bread91)
     (no_gluten_content content125)
     (allergic_gluten child78)
     (not_allergic_gluten child160)
     (waiting child78 table94)
     (waiting child160 table78)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child78)
     (served child160)
    )
  )
)
