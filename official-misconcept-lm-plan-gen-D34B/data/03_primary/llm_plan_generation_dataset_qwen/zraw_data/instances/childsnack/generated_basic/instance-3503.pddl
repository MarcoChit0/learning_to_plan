; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 82562

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child125 child108 - child
    bread165 bread109 - bread-portion
    content171 content342 - content-portion
    tray464 tray282 - tray
    table475 table367 table400 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread109)
     (at_kitchen_content content171)
     (at_kitchen_content content342)
     (no_gluten_bread bread109)
     (no_gluten_content content342)
     (allergic_gluten child108)
     (not_allergic_gluten child125)
     (waiting child125 table400)
     (waiting child108 table367)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child125)
     (served child108)
    )
  )
)
