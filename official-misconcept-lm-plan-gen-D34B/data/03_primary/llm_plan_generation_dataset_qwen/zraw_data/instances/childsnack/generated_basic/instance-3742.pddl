; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 109816

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 - child
    bread86 - bread-portion
    content152 - content-portion
    tray331 tray12 tray379 - tray
    table127 table123 table136 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray331 kitchen)
     (at tray12 kitchen)
     (at tray379 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_content content152)
     (not_allergic_gluten child388)
     (waiting child388 table123)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child388)
    )
  )
)
