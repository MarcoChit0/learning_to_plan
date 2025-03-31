; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 674637

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 - child
    bread165 - bread-portion
    content143 - content-portion
    tray113 tray133 - tray
    table232 table398 table422 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray113 kitchen)
     (at tray133 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_content content143)
     (not_allergic_gluten child492)
     (waiting child492 table422)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child492)
    )
  )
)
