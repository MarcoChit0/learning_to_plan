; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 537067

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child380 child433 - child
    bread41 bread404 - bread-portion
    content317 content143 - content-portion
    tray3 tray492 - tray
    table471 table373 table143 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at tray492 kitchen)
     (at_kitchen_bread bread41)
     (at_kitchen_bread bread404)
     (at_kitchen_content content317)
     (at_kitchen_content content143)
     (no_gluten_bread bread404)
     (no_gluten_content content317)
     (allergic_gluten child433)
     (not_allergic_gluten child380)
     (waiting child380 table143)
     (waiting child433 table143)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child380)
     (served child433)
    )
  )
)
