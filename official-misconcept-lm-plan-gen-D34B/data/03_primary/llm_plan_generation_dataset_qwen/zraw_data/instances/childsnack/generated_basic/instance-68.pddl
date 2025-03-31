; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 276298

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child61 - child
    bread165 bread228 - bread-portion
    content252 content417 - content-portion
    tray378 tray130 - tray
    table69 table203 table326 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray378 kitchen)
     (at tray130 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread228)
     (at_kitchen_content content252)
     (at_kitchen_content content417)
     (not_allergic_gluten child61)
     (not_allergic_gluten child218)
     (waiting child218 table69)
     (waiting child61 table326)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child218)
     (served child61)
    )
  )
)
