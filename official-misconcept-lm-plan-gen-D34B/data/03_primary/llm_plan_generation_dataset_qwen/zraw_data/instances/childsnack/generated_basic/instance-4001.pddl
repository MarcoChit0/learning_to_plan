; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 444301

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread148 - bread-portion
    content387 - content-portion
    tray370 tray309 tray117 - tray
    table377 table310 table368 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at tray309 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_content content387)
     (not_allergic_gluten child431)
     (waiting child431 table377)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
