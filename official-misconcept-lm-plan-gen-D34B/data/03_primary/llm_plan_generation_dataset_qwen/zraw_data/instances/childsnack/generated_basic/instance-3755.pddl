; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 540270

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child40 - child
    bread26 - bread-portion
    content236 - content-portion
    tray320 tray268 tray388 - tray
    table460 table189 table390 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at tray268 kitchen)
     (at tray388 kitchen)
     (at_kitchen_bread bread26)
     (at_kitchen_content content236)
     (not_allergic_gluten child40)
     (waiting child40 table189)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child40)
    )
  )
)
