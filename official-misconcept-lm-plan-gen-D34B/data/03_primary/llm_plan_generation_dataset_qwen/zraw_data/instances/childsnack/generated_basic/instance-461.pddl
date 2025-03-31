; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 442243

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child52 child378 child352 - child
    bread384 bread174 bread266 - bread-portion
    content269 content333 content68 - content-portion
    tray358 - tray
    table420 table382 table383 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at_kitchen_bread bread384)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread266)
     (at_kitchen_content content269)
     (at_kitchen_content content333)
     (at_kitchen_content content68)
     (not_allergic_gluten child52)
     (not_allergic_gluten child378)
     (not_allergic_gluten child352)
     (waiting child52 table383)
     (waiting child378 table420)
     (waiting child352 table382)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child52)
     (served child378)
     (served child352)
    )
  )
)
