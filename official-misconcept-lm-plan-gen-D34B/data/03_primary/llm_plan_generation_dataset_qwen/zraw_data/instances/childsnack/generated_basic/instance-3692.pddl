; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 495010

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 child203 - child
    bread17 bread431 - bread-portion
    content2 content496 - content-portion
    tray432 tray38 - tray
    table153 table358 table436 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray432 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_bread bread431)
     (at_kitchen_content content2)
     (at_kitchen_content content496)
     (no_gluten_bread bread17)
     (no_gluten_content content2)
     (allergic_gluten child203)
     (not_allergic_gluten child283)
     (waiting child283 table153)
     (waiting child203 table153)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child283)
     (served child203)
    )
  )
)
