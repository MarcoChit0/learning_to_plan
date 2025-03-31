; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 787485

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 child461 - child
    bread329 bread5 - bread-portion
    content467 content227 - content-portion
    tray334 - tray
    table163 table380 table238 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at_kitchen_bread bread329)
     (at_kitchen_bread bread5)
     (at_kitchen_content content467)
     (at_kitchen_content content227)
     (not_allergic_gluten child167)
     (not_allergic_gluten child461)
     (waiting child167 table163)
     (waiting child461 table163)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child167)
     (served child461)
    )
  )
)
