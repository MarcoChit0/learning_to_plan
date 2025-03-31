; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 664778

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child15 - child
    bread50 bread370 - bread-portion
    content262 content13 - content-portion
    tray7 tray226 - tray
    table472 table152 table249 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray7 kitchen)
     (at tray226 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_bread bread370)
     (at_kitchen_content content262)
     (at_kitchen_content content13)
     (no_gluten_bread bread50)
     (no_gluten_content content13)
     (allergic_gluten child15)
     (not_allergic_gluten child138)
     (waiting child138 table152)
     (waiting child15 table249)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child138)
     (served child15)
    )
  )
)
