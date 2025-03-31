; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 698031

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 child51 child127 - child
    bread134 bread324 bread405 - bread-portion
    content219 content12 content367 - content-portion
    tray261 tray150 tray216 tray181 - tray
    table82 table109 table487 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray150 kitchen)
     (at tray216 kitchen)
     (at tray181 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread405)
     (at_kitchen_content content219)
     (at_kitchen_content content12)
     (at_kitchen_content content367)
     (not_allergic_gluten child127)
     (not_allergic_gluten child51)
     (not_allergic_gluten child378)
     (waiting child378 table109)
     (waiting child51 table109)
     (waiting child127 table109)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child378)
     (served child51)
     (served child127)
    )
  )
)
