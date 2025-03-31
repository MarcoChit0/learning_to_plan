; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 174413

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child403 child410 - child
    bread456 bread491 - bread-portion
    content375 content462 - content-portion
    tray414 tray233 tray98 tray79 - tray
    table395 table327 table482 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray233 kitchen)
     (at tray98 kitchen)
     (at tray79 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_bread bread491)
     (at_kitchen_content content375)
     (at_kitchen_content content462)
     (no_gluten_bread bread491)
     (no_gluten_content content375)
     (allergic_gluten child410)
     (not_allergic_gluten child403)
     (waiting child403 table395)
     (waiting child410 table327)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child403)
     (served child410)
    )
  )
)
