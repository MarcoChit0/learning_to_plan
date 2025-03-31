; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 383619

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 child231 - child
    bread79 bread177 - bread-portion
    content145 content290 - content-portion
    tray331 tray382 tray174 tray441 - tray
    table390 table328 table355 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray331 kitchen)
     (at tray382 kitchen)
     (at tray174 kitchen)
     (at tray441 kitchen)
     (at_kitchen_bread bread79)
     (at_kitchen_bread bread177)
     (at_kitchen_content content145)
     (at_kitchen_content content290)
     (not_allergic_gluten child231)
     (not_allergic_gluten child59)
     (waiting child59 table355)
     (waiting child231 table328)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child59)
     (served child231)
    )
  )
)
