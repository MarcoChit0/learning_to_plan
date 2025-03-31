; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 741515

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child35 - child
    bread364 bread432 - bread-portion
    content417 content383 - content-portion
    tray308 - tray
    table202 table61 table172 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread432)
     (at_kitchen_content content417)
     (at_kitchen_content content383)
     (no_gluten_bread bread364)
     (no_gluten_content content417)
     (allergic_gluten child35)
     (not_allergic_gluten child144)
     (waiting child144 table172)
     (waiting child35 table202)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child144)
     (served child35)
    )
  )
)
