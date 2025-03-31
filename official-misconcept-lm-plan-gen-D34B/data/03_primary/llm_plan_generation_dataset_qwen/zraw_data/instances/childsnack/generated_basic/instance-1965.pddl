; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 517475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child342 child216 - child
    bread283 bread470 - bread-portion
    content346 content407 - content-portion
    tray271 - tray
    table69 table480 table31 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_bread bread470)
     (at_kitchen_content content346)
     (at_kitchen_content content407)
     (no_gluten_bread bread470)
     (no_gluten_content content407)
     (allergic_gluten child342)
     (not_allergic_gluten child216)
     (waiting child342 table69)
     (waiting child216 table31)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child342)
     (served child216)
    )
  )
)
