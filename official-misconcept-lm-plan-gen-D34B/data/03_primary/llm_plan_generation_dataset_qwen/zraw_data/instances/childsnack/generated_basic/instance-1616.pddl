; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 915802

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 - child
    bread108 - bread-portion
    content457 - content-portion
    tray11 tray394 tray411 tray361 - tray
    table496 table229 table18 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray11 kitchen)
     (at tray394 kitchen)
     (at tray411 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread108)
     (at_kitchen_content content457)
     (not_allergic_gluten child163)
     (waiting child163 table496)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child163)
    )
  )
)
