; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 197113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child296 - child
    bread193 bread266 - bread-portion
    content259 content6 - content-portion
    tray102 tray306 - tray
    table83 table334 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray102 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread266)
     (at_kitchen_content content259)
     (at_kitchen_content content6)
     (not_allergic_gluten child296)
     (not_allergic_gluten child462)
     (waiting child462 table243)
     (waiting child296 table243)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child462)
     (served child296)
    )
  )
)
