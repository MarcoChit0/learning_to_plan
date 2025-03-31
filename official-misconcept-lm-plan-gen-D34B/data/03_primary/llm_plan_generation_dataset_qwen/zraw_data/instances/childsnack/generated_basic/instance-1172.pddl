; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 522372

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child231 - child
    bread289 bread34 - bread-portion
    content111 content87 - content-portion
    tray17 tray452 tray353 - tray
    table434 table121 table310 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray17 kitchen)
     (at tray452 kitchen)
     (at tray353 kitchen)
     (at_kitchen_bread bread289)
     (at_kitchen_bread bread34)
     (at_kitchen_content content111)
     (at_kitchen_content content87)
     (no_gluten_bread bread34)
     (no_gluten_content content111)
     (allergic_gluten child323)
     (not_allergic_gluten child231)
     (waiting child323 table310)
     (waiting child231 table310)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child323)
     (served child231)
    )
  )
)
