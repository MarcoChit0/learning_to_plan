; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 612397

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 child194 child379 - child
    bread136 bread255 bread435 - bread-portion
    content208 content356 content259 - content-portion
    tray174 - tray
    table87 table136 table411 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread255)
     (at_kitchen_bread bread435)
     (at_kitchen_content content208)
     (at_kitchen_content content356)
     (at_kitchen_content content259)
     (not_allergic_gluten child97)
     (not_allergic_gluten child194)
     (not_allergic_gluten child379)
     (waiting child97 table87)
     (waiting child194 table136)
     (waiting child379 table87)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child97)
     (served child194)
     (served child379)
    )
  )
)
