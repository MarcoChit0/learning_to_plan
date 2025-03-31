; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 172315

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child203 - child
    bread321 bread497 - bread-portion
    content297 content22 - content-portion
    tray471 tray463 - tray
    table142 table386 table89 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray463 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread497)
     (at_kitchen_content content297)
     (at_kitchen_content content22)
     (no_gluten_bread bread497)
     (no_gluten_content content297)
     (allergic_gluten child203)
     (not_allergic_gluten child431)
     (waiting child431 table386)
     (waiting child203 table386)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child431)
     (served child203)
    )
  )
)
