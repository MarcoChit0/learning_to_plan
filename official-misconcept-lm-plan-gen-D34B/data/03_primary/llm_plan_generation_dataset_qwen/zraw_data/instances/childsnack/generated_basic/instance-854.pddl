; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 43828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child99 - child
    bread81 bread246 - bread-portion
    content88 content314 - content-portion
    tray215 tray447 - tray
    table20 table286 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray447 kitchen)
     (at_kitchen_bread bread81)
     (at_kitchen_bread bread246)
     (at_kitchen_content content88)
     (at_kitchen_content content314)
     (no_gluten_bread bread246)
     (no_gluten_content content88)
     (allergic_gluten child64)
     (not_allergic_gluten child99)
     (waiting child64 table20)
     (waiting child99 table217)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child64)
     (served child99)
    )
  )
)
