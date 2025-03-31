; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 193978

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child46 child420 - child
    bread162 bread386 - bread-portion
    content72 content139 - content-portion
    tray145 - tray
    table32 table491 table6 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray145 kitchen)
     (at_kitchen_bread bread162)
     (at_kitchen_bread bread386)
     (at_kitchen_content content72)
     (at_kitchen_content content139)
     (no_gluten_bread bread162)
     (no_gluten_content content139)
     (allergic_gluten child46)
     (not_allergic_gluten child420)
     (waiting child46 table6)
     (waiting child420 table32)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child46)
     (served child420)
    )
  )
)
