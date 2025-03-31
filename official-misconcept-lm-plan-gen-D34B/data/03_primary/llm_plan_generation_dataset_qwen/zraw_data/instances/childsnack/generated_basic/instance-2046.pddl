; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 436253

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 child454 - child
    bread90 bread328 - bread-portion
    content282 content337 - content-portion
    tray16 tray336 tray282 - tray
    table55 table125 table206 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray16 kitchen)
     (at tray336 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread90)
     (at_kitchen_bread bread328)
     (at_kitchen_content content282)
     (at_kitchen_content content337)
     (not_allergic_gluten child454)
     (not_allergic_gluten child284)
     (waiting child284 table55)
     (waiting child454 table206)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child284)
     (served child454)
    )
  )
)
