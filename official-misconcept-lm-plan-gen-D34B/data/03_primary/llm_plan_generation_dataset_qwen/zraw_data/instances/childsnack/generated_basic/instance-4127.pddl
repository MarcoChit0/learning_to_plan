; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 875130

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child466 - child
    bread362 bread142 - bread-portion
    content166 content113 - content-portion
    tray429 tray43 tray118 - tray
    table456 table475 table446 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at tray43 kitchen)
     (at tray118 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_bread bread142)
     (at_kitchen_content content166)
     (at_kitchen_content content113)
     (not_allergic_gluten child73)
     (not_allergic_gluten child466)
     (waiting child73 table456)
     (waiting child466 table475)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child73)
     (served child466)
    )
  )
)
