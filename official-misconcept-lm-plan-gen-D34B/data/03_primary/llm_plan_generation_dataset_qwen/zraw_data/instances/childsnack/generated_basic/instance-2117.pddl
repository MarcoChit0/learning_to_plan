; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 842240

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 child343 - child
    bread37 bread488 - bread-portion
    content266 content404 - content-portion
    tray152 tray43 tray258 - tray
    table390 table395 table118 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray152 kitchen)
     (at tray43 kitchen)
     (at tray258 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread488)
     (at_kitchen_content content266)
     (at_kitchen_content content404)
     (no_gluten_bread bread488)
     (no_gluten_content content266)
     (allergic_gluten child248)
     (not_allergic_gluten child343)
     (waiting child248 table390)
     (waiting child343 table395)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child248)
     (served child343)
    )
  )
)
