; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 19580

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 - child
    bread28 - bread-portion
    content263 - content-portion
    tray39 tray243 tray109 - tray
    table331 table149 table372 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray243 kitchen)
     (at tray109 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_content content263)
     (not_allergic_gluten child367)
     (waiting child367 table331)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child367)
    )
  )
)
