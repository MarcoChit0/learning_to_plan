; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 644416

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child340 - child
    bread353 - bread-portion
    content199 - content-portion
    tray173 tray293 tray101 - tray
    table233 table330 table337 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray173 kitchen)
     (at tray293 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_content content199)
     (not_allergic_gluten child340)
     (waiting child340 table233)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child340)
    )
  )
)
