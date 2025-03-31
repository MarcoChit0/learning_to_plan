; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 237869

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 - child
    bread94 - bread-portion
    content367 - content-portion
    tray469 tray421 tray216 - tray
    table335 table222 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at tray421 kitchen)
     (at tray216 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_content content367)
     (not_allergic_gluten child120)
     (waiting child120 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child120)
    )
  )
)
