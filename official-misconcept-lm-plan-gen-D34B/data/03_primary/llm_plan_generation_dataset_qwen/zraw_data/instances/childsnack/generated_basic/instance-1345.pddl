; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 152312

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 child347 - child
    bread321 bread341 - bread-portion
    content427 content244 - content-portion
    tray117 tray438 tray282 - tray
    table352 table420 table339 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at tray438 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread341)
     (at_kitchen_content content427)
     (at_kitchen_content content244)
     (not_allergic_gluten child332)
     (not_allergic_gluten child347)
     (waiting child332 table339)
     (waiting child347 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child332)
     (served child347)
    )
  )
)
