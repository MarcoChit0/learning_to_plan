; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 166534

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child230 - child
    bread156 bread336 - bread-portion
    content87 content70 - content-portion
    tray136 tray239 - tray
    table270 table244 table299 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray136 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread156)
     (at_kitchen_bread bread336)
     (at_kitchen_content content87)
     (at_kitchen_content content70)
     (not_allergic_gluten child85)
     (not_allergic_gluten child230)
     (waiting child85 table299)
     (waiting child230 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child85)
     (served child230)
    )
  )
)
