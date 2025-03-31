; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 379540

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child154 child29 - child
    bread349 bread220 - bread-portion
    content389 content165 - content-portion
    tray215 tray58 tray27 tray28 - tray
    table317 table316 table163 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray58 kitchen)
     (at tray27 kitchen)
     (at tray28 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread220)
     (at_kitchen_content content389)
     (at_kitchen_content content165)
     (no_gluten_bread bread349)
     (no_gluten_content content389)
     (allergic_gluten child29)
     (not_allergic_gluten child154)
     (waiting child154 table316)
     (waiting child29 table317)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child154)
     (served child29)
    )
  )
)
