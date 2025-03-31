; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 506105

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 - child
    bread260 - bread-portion
    content282 - content-portion
    tray335 - tray
    table125 table0 table140 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_content content282)
     (not_allergic_gluten child137)
     (waiting child137 table0)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child137)
    )
  )
)
