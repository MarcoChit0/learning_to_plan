; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 117301

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child288 - child
    bread112 - bread-portion
    content67 - content-portion
    tray361 tray269 tray388 tray72 - tray
    table316 table478 table29 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray361 kitchen)
     (at tray269 kitchen)
     (at tray388 kitchen)
     (at tray72 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_content content67)
     (not_allergic_gluten child288)
     (waiting child288 table478)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child288)
    )
  )
)
