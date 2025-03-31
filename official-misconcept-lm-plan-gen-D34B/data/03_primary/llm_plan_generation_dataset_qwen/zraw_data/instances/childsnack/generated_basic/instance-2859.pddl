; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 190649

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 - child
    bread445 - bread-portion
    content244 - content-portion
    tray384 tray315 - tray
    table300 table238 table311 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray315 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_content content244)
     (not_allergic_gluten child458)
     (waiting child458 table311)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child458)
    )
  )
)
