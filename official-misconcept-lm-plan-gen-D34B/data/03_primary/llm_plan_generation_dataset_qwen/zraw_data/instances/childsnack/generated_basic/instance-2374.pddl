; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 982676

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child498 child183 - child
    bread419 bread48 - bread-portion
    content165 content429 - content-portion
    tray446 - tray
    table423 table5 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_bread bread48)
     (at_kitchen_content content165)
     (at_kitchen_content content429)
     (not_allergic_gluten child498)
     (not_allergic_gluten child183)
     (waiting child498 table217)
     (waiting child183 table5)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child498)
     (served child183)
    )
  )
)
