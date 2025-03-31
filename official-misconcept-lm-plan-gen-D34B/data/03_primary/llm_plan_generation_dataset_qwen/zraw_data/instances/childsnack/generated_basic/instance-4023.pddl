; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 972521

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child48 child271 child459 - child
    bread63 bread9 bread267 - bread-portion
    content300 content291 content333 - content-portion
    tray124 - tray
    table223 table347 table142 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray124 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread9)
     (at_kitchen_bread bread267)
     (at_kitchen_content content300)
     (at_kitchen_content content291)
     (at_kitchen_content content333)
     (not_allergic_gluten child459)
     (not_allergic_gluten child48)
     (not_allergic_gluten child271)
     (waiting child48 table223)
     (waiting child271 table223)
     (waiting child459 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child48)
     (served child271)
     (served child459)
    )
  )
)
