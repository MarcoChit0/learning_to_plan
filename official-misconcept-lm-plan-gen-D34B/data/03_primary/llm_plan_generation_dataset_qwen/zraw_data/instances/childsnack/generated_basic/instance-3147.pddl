; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 38304

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 child231 - child
    bread337 bread367 - bread-portion
    content161 content129 - content-portion
    tray154 tray65 - tray
    table323 table161 table8 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at tray65 kitchen)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread367)
     (at_kitchen_content content161)
     (at_kitchen_content content129)
     (not_allergic_gluten child231)
     (not_allergic_gluten child486)
     (waiting child486 table161)
     (waiting child231 table8)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child486)
     (served child231)
    )
  )
)
