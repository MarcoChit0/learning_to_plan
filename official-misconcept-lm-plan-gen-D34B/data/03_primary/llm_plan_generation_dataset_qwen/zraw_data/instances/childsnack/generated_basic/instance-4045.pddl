; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 238038

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child237 child399 - child
    bread467 bread94 bread342 - bread-portion
    content136 content73 content51 - content-portion
    tray97 - tray
    table92 table269 table384 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at_kitchen_bread bread467)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread342)
     (at_kitchen_content content136)
     (at_kitchen_content content73)
     (at_kitchen_content content51)
     (not_allergic_gluten child432)
     (not_allergic_gluten child399)
     (not_allergic_gluten child237)
     (waiting child432 table269)
     (waiting child237 table269)
     (waiting child399 table92)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child432)
     (served child237)
     (served child399)
    )
  )
)
