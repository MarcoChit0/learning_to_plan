; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 48383

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 child201 - child
    bread420 bread135 - bread-portion
    content478 content3 - content-portion
    tray60 tray302 - tray
    table435 table40 table285 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray302 kitchen)
     (at_kitchen_bread bread420)
     (at_kitchen_bread bread135)
     (at_kitchen_content content478)
     (at_kitchen_content content3)
     (not_allergic_gluten child7)
     (not_allergic_gluten child201)
     (waiting child7 table435)
     (waiting child201 table285)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child7)
     (served child201)
    )
  )
)
