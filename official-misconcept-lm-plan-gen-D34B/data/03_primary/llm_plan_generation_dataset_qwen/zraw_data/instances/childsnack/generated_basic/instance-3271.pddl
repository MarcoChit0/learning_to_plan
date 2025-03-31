; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 144302

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 child497 - child
    bread459 bread487 - bread-portion
    content282 content106 - content-portion
    tray226 tray63 tray24 tray498 - tray
    table268 table118 table420 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray63 kitchen)
     (at tray24 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread487)
     (at_kitchen_content content282)
     (at_kitchen_content content106)
     (no_gluten_bread bread487)
     (no_gluten_content content282)
     (allergic_gluten child460)
     (not_allergic_gluten child497)
     (waiting child460 table420)
     (waiting child497 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child460)
     (served child497)
    )
  )
)
