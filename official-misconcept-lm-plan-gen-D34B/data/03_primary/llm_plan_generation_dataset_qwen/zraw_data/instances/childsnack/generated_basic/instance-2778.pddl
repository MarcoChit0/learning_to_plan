; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 438603

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child449 child190 - child
    bread478 bread138 - bread-portion
    content394 content487 - content-portion
    tray375 - tray
    table196 table15 table27 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray375 kitchen)
     (at_kitchen_bread bread478)
     (at_kitchen_bread bread138)
     (at_kitchen_content content394)
     (at_kitchen_content content487)
     (no_gluten_bread bread138)
     (no_gluten_content content487)
     (allergic_gluten child190)
     (not_allergic_gluten child449)
     (waiting child449 table15)
     (waiting child190 table27)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child449)
     (served child190)
    )
  )
)
