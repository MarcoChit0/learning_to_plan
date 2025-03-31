; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 681941

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 child446 - child
    bread26 bread322 - bread-portion
    content201 content30 - content-portion
    tray347 tray288 tray170 tray140 - tray
    table478 table144 table370 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray347 kitchen)
     (at tray288 kitchen)
     (at tray170 kitchen)
     (at tray140 kitchen)
     (at_kitchen_bread bread26)
     (at_kitchen_bread bread322)
     (at_kitchen_content content201)
     (at_kitchen_content content30)
     (no_gluten_bread bread322)
     (no_gluten_content content201)
     (allergic_gluten child398)
     (not_allergic_gluten child446)
     (waiting child398 table144)
     (waiting child446 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child398)
     (served child446)
    )
  )
)
