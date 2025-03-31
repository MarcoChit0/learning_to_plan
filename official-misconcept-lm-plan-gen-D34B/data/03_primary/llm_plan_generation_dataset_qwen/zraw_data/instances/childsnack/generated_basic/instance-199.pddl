; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 242934

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child45 - child
    bread304 bread142 - bread-portion
    content426 content403 - content-portion
    tray178 tray291 tray393 - tray
    table172 table491 table344 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray178 kitchen)
     (at tray291 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_bread bread142)
     (at_kitchen_content content426)
     (at_kitchen_content content403)
     (not_allergic_gluten child112)
     (not_allergic_gluten child45)
     (waiting child112 table172)
     (waiting child45 table491)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child112)
     (served child45)
    )
  )
)
