; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 969183

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 child288 child390 - child
    bread481 bread53 bread282 - bread-portion
    content5 content350 content387 - content-portion
    tray101 tray345 tray402 - tray
    table244 table121 table332 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray101 kitchen)
     (at tray345 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_bread bread53)
     (at_kitchen_bread bread282)
     (at_kitchen_content content5)
     (at_kitchen_content content350)
     (at_kitchen_content content387)
     (not_allergic_gluten child441)
     (not_allergic_gluten child288)
     (not_allergic_gluten child390)
     (waiting child441 table121)
     (waiting child288 table332)
     (waiting child390 table332)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child441)
     (served child288)
     (served child390)
    )
  )
)
