; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.7
; random seed: 156836

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child58 child138 - child
    bread311 bread417 - bread-portion
    content369 content339 - content-portion
    tray201 tray344 tray85 - tray
    table403 table71 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray201 kitchen)
     (at tray344 kitchen)
     (at tray85 kitchen)
     (at_kitchen_bread bread311)
     (at_kitchen_bread bread417)
     (at_kitchen_content content369)
     (at_kitchen_content content339)
     (no_gluten_bread bread311)
     (no_gluten_content content339)
     (allergic_gluten child58)
     (not_allergic_gluten child138)
     (waiting child58 table71)
     (waiting child138 table477)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child58)
     (served child138)
    )
  )
)
