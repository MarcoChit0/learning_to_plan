; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 826489

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 child220 - child
    bread42 bread349 - bread-portion
    content48 content197 - content-portion
    tray186 - tray
    table161 table105 table302 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread349)
     (at_kitchen_content content48)
     (at_kitchen_content content197)
     (no_gluten_bread bread349)
     (no_gluten_content content197)
     (allergic_gluten child220)
     (not_allergic_gluten child30)
     (waiting child30 table161)
     (waiting child220 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child30)
     (served child220)
    )
  )
)
