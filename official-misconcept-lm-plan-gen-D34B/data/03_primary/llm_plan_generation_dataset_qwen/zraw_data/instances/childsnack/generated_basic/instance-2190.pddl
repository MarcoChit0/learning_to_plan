; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 942523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 - child
    bread61 - bread-portion
    content212 - content-portion
    tray333 - tray
    table129 table163 table165 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at_kitchen_bread bread61)
     (at_kitchen_content content212)
     (not_allergic_gluten child399)
     (waiting child399 table165)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child399)
    )
  )
)
