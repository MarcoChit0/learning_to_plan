; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 191221

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child46 child221 - child
    bread25 bread425 - bread-portion
    content390 content354 - content-portion
    tray49 tray86 - tray
    table12 table70 table471 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread25)
     (at_kitchen_bread bread425)
     (at_kitchen_content content390)
     (at_kitchen_content content354)
     (not_allergic_gluten child221)
     (not_allergic_gluten child46)
     (waiting child46 table70)
     (waiting child221 table471)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child46)
     (served child221)
    )
  )
)
