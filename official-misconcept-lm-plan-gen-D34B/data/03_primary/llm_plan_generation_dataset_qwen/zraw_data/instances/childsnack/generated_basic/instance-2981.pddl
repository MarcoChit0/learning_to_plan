; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 359273

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 child45 - child
    bread7 bread6 - bread-portion
    content16 content203 - content-portion
    tray148 tray209 tray272 tray106 - tray
    table343 table450 table330 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray148 kitchen)
     (at tray209 kitchen)
     (at tray272 kitchen)
     (at tray106 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread6)
     (at_kitchen_content content16)
     (at_kitchen_content content203)
     (not_allergic_gluten child45)
     (not_allergic_gluten child325)
     (waiting child325 table450)
     (waiting child45 table450)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child325)
     (served child45)
    )
  )
)
