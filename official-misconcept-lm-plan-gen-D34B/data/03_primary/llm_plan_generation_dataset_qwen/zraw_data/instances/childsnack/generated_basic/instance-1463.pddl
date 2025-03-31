; child-snack task with 3 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 262392

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child293 child315 - child
    bread405 bread426 bread376 - bread-portion
    content289 content412 content199 - content-portion
    tray182 - tray
    table312 table251 table429 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at_kitchen_bread bread405)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread376)
     (at_kitchen_content content289)
     (at_kitchen_content content412)
     (at_kitchen_content content199)
     (no_gluten_bread bread426)
     (no_gluten_bread bread405)
     (no_gluten_content content412)
     (no_gluten_content content289)
     (allergic_gluten child200)
     (allergic_gluten child315)
     (not_allergic_gluten child293)
     (waiting child200 table429)
     (waiting child293 table251)
     (waiting child315 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child200)
     (served child293)
     (served child315)
    )
  )
)
