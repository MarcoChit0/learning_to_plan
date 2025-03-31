; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 895409

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child70 child338 child383 - child
    bread394 bread133 bread344 - bread-portion
    content69 content166 content346 - content-portion
    tray200 tray463 tray411 - tray
    table280 table478 table107 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray463 kitchen)
     (at tray411 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread133)
     (at_kitchen_bread bread344)
     (at_kitchen_content content69)
     (at_kitchen_content content166)
     (at_kitchen_content content346)
     (not_allergic_gluten child383)
     (not_allergic_gluten child70)
     (not_allergic_gluten child338)
     (waiting child70 table280)
     (waiting child338 table478)
     (waiting child383 table107)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child70)
     (served child338)
     (served child383)
    )
  )
)
