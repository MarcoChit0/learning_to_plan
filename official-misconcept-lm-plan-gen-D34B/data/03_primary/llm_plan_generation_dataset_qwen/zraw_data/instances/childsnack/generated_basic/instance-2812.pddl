; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 842943

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child269 - child
    bread391 bread225 - bread-portion
    content322 content143 - content-portion
    tray135 tray493 - tray
    table297 table67 table407 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray135 kitchen)
     (at tray493 kitchen)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread225)
     (at_kitchen_content content322)
     (at_kitchen_content content143)
     (not_allergic_gluten child269)
     (not_allergic_gluten child237)
     (waiting child237 table407)
     (waiting child269 table297)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child237)
     (served child269)
    )
  )
)
