; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 361890

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 child211 child304 - child
    bread348 bread193 bread58 - bread-portion
    content393 content286 content32 - content-portion
    tray24 - tray
    table9 table342 table120 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread58)
     (at_kitchen_content content393)
     (at_kitchen_content content286)
     (at_kitchen_content content32)
     (not_allergic_gluten child304)
     (not_allergic_gluten child474)
     (not_allergic_gluten child211)
     (waiting child474 table342)
     (waiting child211 table120)
     (waiting child304 table9)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child474)
     (served child211)
     (served child304)
    )
  )
)
