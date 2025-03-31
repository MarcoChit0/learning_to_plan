; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 811638

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child393 - child
    bread464 bread492 - bread-portion
    content423 content29 - content-portion
    tray87 tray474 - tray
    table489 table342 table348 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray87 kitchen)
     (at tray474 kitchen)
     (at_kitchen_bread bread464)
     (at_kitchen_bread bread492)
     (at_kitchen_content content423)
     (at_kitchen_content content29)
     (not_allergic_gluten child393)
     (not_allergic_gluten child246)
     (waiting child246 table489)
     (waiting child393 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child246)
     (served child393)
    )
  )
)
