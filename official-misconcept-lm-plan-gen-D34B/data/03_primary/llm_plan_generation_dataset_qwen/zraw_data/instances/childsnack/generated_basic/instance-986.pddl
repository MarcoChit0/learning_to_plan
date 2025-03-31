; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 265031

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child366 child292 child331 - child
    bread349 bread485 bread111 - bread-portion
    content189 content193 content402 - content-portion
    tray143 - tray
    table405 table377 table178 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray143 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread485)
     (at_kitchen_bread bread111)
     (at_kitchen_content content189)
     (at_kitchen_content content193)
     (at_kitchen_content content402)
     (not_allergic_gluten child331)
     (not_allergic_gluten child292)
     (not_allergic_gluten child366)
     (waiting child366 table178)
     (waiting child292 table178)
     (waiting child331 table377)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child366)
     (served child292)
     (served child331)
    )
  )
)
