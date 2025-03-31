; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 969539

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 - child
    bread426 - bread-portion
    content422 - content-portion
    tray449 tray487 tray340 tray406 - tray
    table45 table445 table129 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray487 kitchen)
     (at tray340 kitchen)
     (at tray406 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_content content422)
     (not_allergic_gluten child447)
     (waiting child447 table445)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child447)
    )
  )
)
