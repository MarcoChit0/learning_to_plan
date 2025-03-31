; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 883628

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child28 - child
    bread4 bread230 - bread-portion
    content6 content93 - content-portion
    tray234 tray381 tray137 tray183 - tray
    table10 table158 table488 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray381 kitchen)
     (at tray137 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread230)
     (at_kitchen_content content6)
     (at_kitchen_content content93)
     (not_allergic_gluten child28)
     (not_allergic_gluten child442)
     (waiting child442 table10)
     (waiting child28 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child28)
    )
  )
)
