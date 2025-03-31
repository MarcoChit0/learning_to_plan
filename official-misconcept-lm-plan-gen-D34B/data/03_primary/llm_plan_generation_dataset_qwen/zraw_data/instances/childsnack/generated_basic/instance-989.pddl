; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 809621

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 child473 child272 - child
    bread260 bread304 bread44 - bread-portion
    content489 content346 content28 - content-portion
    tray233 - tray
    table348 table72 table274 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_bread bread304)
     (at_kitchen_bread bread44)
     (at_kitchen_content content489)
     (at_kitchen_content content346)
     (at_kitchen_content content28)
     (not_allergic_gluten child473)
     (not_allergic_gluten child168)
     (not_allergic_gluten child272)
     (waiting child168 table274)
     (waiting child473 table348)
     (waiting child272 table72)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child168)
     (served child473)
     (served child272)
    )
  )
)
