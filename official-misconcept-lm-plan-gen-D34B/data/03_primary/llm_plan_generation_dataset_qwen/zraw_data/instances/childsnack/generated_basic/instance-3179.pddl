; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 918695

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 child62 - child
    bread297 bread240 - bread-portion
    content312 content280 - content-portion
    tray186 tray271 - tray
    table189 table81 table144 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread240)
     (at_kitchen_content content312)
     (at_kitchen_content content280)
     (not_allergic_gluten child62)
     (not_allergic_gluten child304)
     (waiting child304 table81)
     (waiting child62 table189)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child304)
     (served child62)
    )
  )
)
