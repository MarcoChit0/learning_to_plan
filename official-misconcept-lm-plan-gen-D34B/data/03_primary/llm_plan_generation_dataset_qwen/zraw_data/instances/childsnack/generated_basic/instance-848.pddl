; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 99660

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 child5 - child
    bread411 bread96 - bread-portion
    content431 content358 - content-portion
    tray300 tray154 - tray
    table309 table492 table131 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray300 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread411)
     (at_kitchen_bread bread96)
     (at_kitchen_content content431)
     (at_kitchen_content content358)
     (no_gluten_bread bread96)
     (no_gluten_content content431)
     (allergic_gluten child43)
     (not_allergic_gluten child5)
     (waiting child43 table492)
     (waiting child5 table131)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child43)
     (served child5)
    )
  )
)
