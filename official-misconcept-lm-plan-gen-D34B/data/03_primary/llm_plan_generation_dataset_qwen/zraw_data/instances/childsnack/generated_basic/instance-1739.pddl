; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 896323

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread207 - bread-portion
    content68 - content-portion
    tray287 tray273 tray7 - tray
    table362 table224 table159 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray287 kitchen)
     (at tray273 kitchen)
     (at tray7 kitchen)
     (at_kitchen_bread bread207)
     (at_kitchen_content content68)
     (not_allergic_gluten child8)
     (waiting child8 table362)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
