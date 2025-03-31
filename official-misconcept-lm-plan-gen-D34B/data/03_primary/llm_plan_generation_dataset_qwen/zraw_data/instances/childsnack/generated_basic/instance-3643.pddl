; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 222527

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 child106 - child
    bread196 bread257 - bread-portion
    content8 content355 - content-portion
    tray167 tray116 tray261 tray200 - tray
    table493 table5 table321 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray116 kitchen)
     (at tray261 kitchen)
     (at tray200 kitchen)
     (at_kitchen_bread bread196)
     (at_kitchen_bread bread257)
     (at_kitchen_content content8)
     (at_kitchen_content content355)
     (no_gluten_bread bread257)
     (no_gluten_content content355)
     (allergic_gluten child196)
     (not_allergic_gluten child106)
     (waiting child196 table5)
     (waiting child106 table5)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child196)
     (served child106)
    )
  )
)
