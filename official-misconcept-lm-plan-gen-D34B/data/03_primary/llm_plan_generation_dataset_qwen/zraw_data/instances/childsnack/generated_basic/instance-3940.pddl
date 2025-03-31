; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 454283

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child298 child498 - child
    bread259 bread100 - bread-portion
    content407 content150 - content-portion
    tray105 - tray
    table20 table134 table37 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread100)
     (at_kitchen_content content407)
     (at_kitchen_content content150)
     (no_gluten_bread bread259)
     (no_gluten_content content150)
     (allergic_gluten child298)
     (not_allergic_gluten child498)
     (waiting child298 table20)
     (waiting child498 table134)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child298)
     (served child498)
    )
  )
)
