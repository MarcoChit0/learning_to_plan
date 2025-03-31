; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 529247

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 child134 child1 - child
    bread71 bread244 bread8 - bread-portion
    content416 content89 content345 - content-portion
    tray277 - tray
    table246 table19 table384 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_bread bread244)
     (at_kitchen_bread bread8)
     (at_kitchen_content content416)
     (at_kitchen_content content89)
     (at_kitchen_content content345)
     (not_allergic_gluten child134)
     (not_allergic_gluten child248)
     (not_allergic_gluten child1)
     (waiting child248 table246)
     (waiting child134 table384)
     (waiting child1 table384)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child248)
     (served child134)
     (served child1)
    )
  )
)
