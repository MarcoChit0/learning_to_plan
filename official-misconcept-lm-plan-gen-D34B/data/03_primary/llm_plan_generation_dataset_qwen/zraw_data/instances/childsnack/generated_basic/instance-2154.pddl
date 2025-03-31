; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 829535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child299 child164 - child
    bread217 bread329 - bread-portion
    content436 content289 - content-portion
    tray54 tray247 tray229 - tray
    table192 table166 table172 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at tray247 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread217)
     (at_kitchen_bread bread329)
     (at_kitchen_content content436)
     (at_kitchen_content content289)
     (no_gluten_bread bread217)
     (no_gluten_content content436)
     (allergic_gluten child299)
     (not_allergic_gluten child164)
     (waiting child299 table166)
     (waiting child164 table172)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child299)
     (served child164)
    )
  )
)
