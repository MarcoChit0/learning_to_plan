; child-snack task with 3 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 441412

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child92 child137 - child
    bread310 bread407 bread312 - bread-portion
    content59 content272 content493 - content-portion
    tray13 - tray
    table331 table209 table173 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray13 kitchen)
     (at_kitchen_bread bread310)
     (at_kitchen_bread bread407)
     (at_kitchen_bread bread312)
     (at_kitchen_content content59)
     (at_kitchen_content content272)
     (at_kitchen_content content493)
     (no_gluten_bread bread407)
     (no_gluten_bread bread310)
     (no_gluten_content content493)
     (no_gluten_content content59)
     (allergic_gluten child225)
     (allergic_gluten child92)
     (not_allergic_gluten child137)
     (waiting child225 table173)
     (waiting child92 table173)
     (waiting child137 table173)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child225)
     (served child92)
     (served child137)
    )
  )
)
