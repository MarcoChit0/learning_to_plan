; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 53796

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 child376 - child
    bread268 bread132 - bread-portion
    content360 content284 - content-portion
    tray442 tray436 tray95 - tray
    table284 table484 table129 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray442 kitchen)
     (at tray436 kitchen)
     (at tray95 kitchen)
     (at_kitchen_bread bread268)
     (at_kitchen_bread bread132)
     (at_kitchen_content content360)
     (at_kitchen_content content284)
     (not_allergic_gluten child33)
     (not_allergic_gluten child376)
     (waiting child33 table484)
     (waiting child376 table484)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child33)
     (served child376)
    )
  )
)
