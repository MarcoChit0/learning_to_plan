; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 269163

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 - child
    bread316 - bread-portion
    content7 - content-portion
    tray436 tray482 tray404 tray222 - tray
    table50 table157 table341 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray482 kitchen)
     (at tray404 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_content content7)
     (not_allergic_gluten child66)
     (waiting child66 table50)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child66)
    )
  )
)
