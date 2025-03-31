; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 696828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 child85 - child
    bread427 bread181 - bread-portion
    content145 content265 - content-portion
    tray251 tray229 - tray
    table332 table15 table222 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray251 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread427)
     (at_kitchen_bread bread181)
     (at_kitchen_content content145)
     (at_kitchen_content content265)
     (not_allergic_gluten child85)
     (not_allergic_gluten child355)
     (waiting child355 table15)
     (waiting child85 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child355)
     (served child85)
    )
  )
)
