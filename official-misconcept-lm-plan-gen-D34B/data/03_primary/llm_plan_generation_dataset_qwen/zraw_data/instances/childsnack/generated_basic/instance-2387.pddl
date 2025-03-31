; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 167513

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child329 - child
    bread306 bread78 - bread-portion
    content90 content389 - content-portion
    tray117 - tray
    table365 table269 table297 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread78)
     (at_kitchen_content content90)
     (at_kitchen_content content389)
     (not_allergic_gluten child308)
     (not_allergic_gluten child329)
     (waiting child308 table297)
     (waiting child329 table269)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child308)
     (served child329)
    )
  )
)
