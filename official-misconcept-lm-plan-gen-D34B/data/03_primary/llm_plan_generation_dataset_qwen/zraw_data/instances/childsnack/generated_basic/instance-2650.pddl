; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 891431

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child242 - child
    bread356 - bread-portion
    content248 - content-portion
    tray167 tray341 tray415 tray256 - tray
    table259 table92 table54 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray341 kitchen)
     (at tray415 kitchen)
     (at tray256 kitchen)
     (at_kitchen_bread bread356)
     (at_kitchen_content content248)
     (not_allergic_gluten child242)
     (waiting child242 table92)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child242)
    )
  )
)
