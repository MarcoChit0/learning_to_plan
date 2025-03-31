; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 389650

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child402 child12 - child
    bread196 bread339 - bread-portion
    content27 content271 - content-portion
    tray68 tray8 tray324 - tray
    table2 table236 table193 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at tray8 kitchen)
     (at tray324 kitchen)
     (at_kitchen_bread bread196)
     (at_kitchen_bread bread339)
     (at_kitchen_content content27)
     (at_kitchen_content content271)
     (not_allergic_gluten child402)
     (not_allergic_gluten child12)
     (waiting child402 table2)
     (waiting child12 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child402)
     (served child12)
    )
  )
)
