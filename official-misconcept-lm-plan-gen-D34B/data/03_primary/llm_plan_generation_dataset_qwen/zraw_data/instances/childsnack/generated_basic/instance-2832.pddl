; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 186182

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child11 - child
    bread261 bread226 - bread-portion
    content473 content448 - content-portion
    tray290 tray115 - tray
    table347 table460 table111 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray290 kitchen)
     (at tray115 kitchen)
     (at_kitchen_bread bread261)
     (at_kitchen_bread bread226)
     (at_kitchen_content content473)
     (at_kitchen_content content448)
     (not_allergic_gluten child11)
     (not_allergic_gluten child362)
     (waiting child362 table460)
     (waiting child11 table460)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child11)
    )
  )
)
