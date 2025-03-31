; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 207080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 - child
    bread96 - bread-portion
    content162 - content-portion
    tray273 tray174 - tray
    table69 table410 table331 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray273 kitchen)
     (at tray174 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_content content162)
     (not_allergic_gluten child43)
     (waiting child43 table69)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child43)
    )
  )
)
