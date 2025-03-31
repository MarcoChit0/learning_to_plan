; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 340849

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 child81 - child
    bread167 bread129 - bread-portion
    content397 content32 - content-portion
    tray422 tray313 - tray
    table132 table201 table111 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray313 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread129)
     (at_kitchen_content content397)
     (at_kitchen_content content32)
     (no_gluten_bread bread167)
     (no_gluten_content content397)
     (allergic_gluten child333)
     (not_allergic_gluten child81)
     (waiting child333 table132)
     (waiting child81 table111)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child333)
     (served child81)
    )
  )
)
