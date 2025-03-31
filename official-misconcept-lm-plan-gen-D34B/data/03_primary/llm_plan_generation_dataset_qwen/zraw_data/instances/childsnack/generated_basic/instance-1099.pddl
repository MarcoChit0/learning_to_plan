; child-snack task with 3 children and 0.38 gluten factor 
; constant factor of 1.5
; random seed: 456554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 child151 child212 - child
    bread12 bread31 bread447 - bread-portion
    content73 content5 content365 - content-portion
    tray285 tray385 tray28 - tray
    table40 table329 table153 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray385 kitchen)
     (at tray28 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_bread bread31)
     (at_kitchen_bread bread447)
     (at_kitchen_content content73)
     (at_kitchen_content content5)
     (at_kitchen_content content365)
     (no_gluten_bread bread12)
     (no_gluten_content content5)
     (allergic_gluten child212)
     (not_allergic_gluten child53)
     (not_allergic_gluten child151)
     (waiting child53 table40)
     (waiting child151 table153)
     (waiting child212 table329)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child53)
     (served child151)
     (served child212)
    )
  )
)
