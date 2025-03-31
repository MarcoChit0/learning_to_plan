; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 575542

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 child120 - child
    bread123 bread322 - bread-portion
    content121 content380 - content-portion
    tray20 tray243 tray471 tray221 - tray
    table38 table6 table108 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at tray243 kitchen)
     (at tray471 kitchen)
     (at tray221 kitchen)
     (at_kitchen_bread bread123)
     (at_kitchen_bread bread322)
     (at_kitchen_content content121)
     (at_kitchen_content content380)
     (no_gluten_bread bread322)
     (no_gluten_content content121)
     (allergic_gluten child120)
     (not_allergic_gluten child229)
     (waiting child229 table6)
     (waiting child120 table38)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child229)
     (served child120)
    )
  )
)
