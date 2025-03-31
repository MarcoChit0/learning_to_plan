; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 437923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child16 - child
    bread78 bread485 - bread-portion
    content415 content23 - content-portion
    tray408 tray59 tray112 - tray
    table437 table199 table317 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at tray59 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_bread bread485)
     (at_kitchen_content content415)
     (at_kitchen_content content23)
     (no_gluten_bread bread485)
     (no_gluten_content content415)
     (allergic_gluten child431)
     (not_allergic_gluten child16)
     (waiting child431 table199)
     (waiting child16 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child431)
     (served child16)
    )
  )
)
