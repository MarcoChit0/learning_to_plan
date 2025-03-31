; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 908530

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child105 - child
    bread187 bread20 - bread-portion
    content107 content12 - content-portion
    tray79 - tray
    table303 table376 table286 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray79 kitchen)
     (at_kitchen_bread bread187)
     (at_kitchen_bread bread20)
     (at_kitchen_content content107)
     (at_kitchen_content content12)
     (no_gluten_bread bread20)
     (no_gluten_content content12)
     (allergic_gluten child105)
     (not_allergic_gluten child495)
     (waiting child495 table303)
     (waiting child105 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child105)
    )
  )
)
