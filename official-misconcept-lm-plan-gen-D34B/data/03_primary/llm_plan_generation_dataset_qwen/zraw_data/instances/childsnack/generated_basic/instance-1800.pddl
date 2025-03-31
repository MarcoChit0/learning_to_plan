; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 896024

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 - child
    bread442 - bread-portion
    content64 - content-portion
    tray460 - tray
    table369 table166 table92 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray460 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_content content64)
     (not_allergic_gluten child416)
     (waiting child416 table166)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child416)
    )
  )
)
