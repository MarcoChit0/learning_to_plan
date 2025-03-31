; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 179655

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child146 - child
    bread126 - bread-portion
    content209 - content-portion
    tray1 tray458 tray225 tray314 - tray
    table213 table54 table237 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at tray458 kitchen)
     (at tray225 kitchen)
     (at tray314 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_content content209)
     (not_allergic_gluten child146)
     (waiting child146 table54)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child146)
    )
  )
)
