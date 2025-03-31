; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 711137

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child411 child164 - child
    bread480 bread404 - bread-portion
    content290 content126 - content-portion
    tray9 tray349 tray143 tray304 - tray
    table489 table147 table204 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray9 kitchen)
     (at tray349 kitchen)
     (at tray143 kitchen)
     (at tray304 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_bread bread404)
     (at_kitchen_content content290)
     (at_kitchen_content content126)
     (not_allergic_gluten child164)
     (not_allergic_gluten child411)
     (waiting child411 table147)
     (waiting child164 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child411)
     (served child164)
    )
  )
)
