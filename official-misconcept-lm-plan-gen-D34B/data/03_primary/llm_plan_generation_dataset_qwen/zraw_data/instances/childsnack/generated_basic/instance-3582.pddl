; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 903510

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child98 child240 child234 - child
    bread369 bread47 bread414 - bread-portion
    content363 content311 content348 - content-portion
    tray213 tray378 tray466 - tray
    table22 table280 table490 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray213 kitchen)
     (at tray378 kitchen)
     (at tray466 kitchen)
     (at_kitchen_bread bread369)
     (at_kitchen_bread bread47)
     (at_kitchen_bread bread414)
     (at_kitchen_content content363)
     (at_kitchen_content content311)
     (at_kitchen_content content348)
     (no_gluten_bread bread414)
     (no_gluten_content content311)
     (allergic_gluten child98)
     (not_allergic_gluten child234)
     (not_allergic_gluten child240)
     (waiting child98 table22)
     (waiting child240 table22)
     (waiting child234 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child98)
     (served child240)
     (served child234)
    )
  )
)
