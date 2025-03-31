; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 914217

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child457 child128 child453 - child
    bread242 bread37 bread224 - bread-portion
    content355 content343 content469 - content-portion
    tray268 - tray
    table119 table32 table173 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray268 kitchen)
     (at_kitchen_bread bread242)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread224)
     (at_kitchen_content content355)
     (at_kitchen_content content343)
     (at_kitchen_content content469)
     (not_allergic_gluten child453)
     (not_allergic_gluten child128)
     (not_allergic_gluten child457)
     (waiting child457 table173)
     (waiting child128 table119)
     (waiting child453 table173)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child457)
     (served child128)
     (served child453)
    )
  )
)
