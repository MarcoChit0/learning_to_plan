; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 18329

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread218 - bread-portion
    content489 - content-portion
    tray295 tray206 tray77 - tray
    table206 table357 table218 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at tray206 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_content content489)
     (not_allergic_gluten child421)
     (waiting child421 table218)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
