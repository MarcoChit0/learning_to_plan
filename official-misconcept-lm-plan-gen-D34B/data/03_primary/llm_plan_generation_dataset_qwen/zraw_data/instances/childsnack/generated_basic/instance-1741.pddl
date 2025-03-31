; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 634118

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 - child
    bread373 - bread-portion
    content454 - content-portion
    tray326 tray471 tray386 - tray
    table261 table49 table339 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray326 kitchen)
     (at tray471 kitchen)
     (at tray386 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_content content454)
     (not_allergic_gluten child131)
     (waiting child131 table261)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child131)
    )
  )
)
