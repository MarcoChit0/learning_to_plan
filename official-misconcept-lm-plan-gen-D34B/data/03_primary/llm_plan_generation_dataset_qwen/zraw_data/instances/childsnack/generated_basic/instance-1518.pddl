; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 611024

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child438 - child
    bread320 bread345 - bread-portion
    content72 content49 - content-portion
    tray185 tray318 tray217 - tray
    table240 table263 table226 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray318 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_bread bread345)
     (at_kitchen_content content72)
     (at_kitchen_content content49)
     (no_gluten_bread bread345)
     (no_gluten_content content49)
     (allergic_gluten child345)
     (not_allergic_gluten child438)
     (waiting child345 table240)
     (waiting child438 table226)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child345)
     (served child438)
    )
  )
)
