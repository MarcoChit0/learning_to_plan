; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 159410

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child380 child40 child9 - child
    bread268 bread409 bread290 - bread-portion
    content10 content166 content358 - content-portion
    tray297 - tray
    table495 table385 table176 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at_kitchen_bread bread268)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread290)
     (at_kitchen_content content10)
     (at_kitchen_content content166)
     (at_kitchen_content content358)
     (not_allergic_gluten child40)
     (not_allergic_gluten child9)
     (not_allergic_gluten child380)
     (waiting child380 table385)
     (waiting child40 table495)
     (waiting child9 table176)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child380)
     (served child40)
     (served child9)
    )
  )
)
