; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 918257

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child234 child425 - child
    bread317 bread427 - bread-portion
    content442 content146 - content-portion
    tray52 tray343 tray384 tray19 - tray
    table2 table429 table385 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray343 kitchen)
     (at tray384 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_bread bread427)
     (at_kitchen_content content442)
     (at_kitchen_content content146)
     (no_gluten_bread bread317)
     (no_gluten_content content442)
     (allergic_gluten child425)
     (not_allergic_gluten child234)
     (waiting child234 table2)
     (waiting child425 table385)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child234)
     (served child425)
    )
  )
)
