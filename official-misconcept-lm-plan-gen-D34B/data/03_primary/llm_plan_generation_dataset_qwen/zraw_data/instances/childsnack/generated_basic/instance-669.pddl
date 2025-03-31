; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 632256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child290 child400 - child
    bread261 bread103 - bread-portion
    content258 content43 - content-portion
    tray464 tray389 - tray
    table442 table108 table106 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread261)
     (at_kitchen_bread bread103)
     (at_kitchen_content content258)
     (at_kitchen_content content43)
     (no_gluten_bread bread261)
     (no_gluten_content content43)
     (allergic_gluten child400)
     (not_allergic_gluten child290)
     (waiting child290 table442)
     (waiting child400 table106)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child290)
     (served child400)
    )
  )
)
