; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 926832

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child3 child489 - child
    bread464 bread40 - bread-portion
    content463 content188 - content-portion
    tray166 tray143 tray375 tray134 - tray
    table118 table388 table69 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at tray143 kitchen)
     (at tray375 kitchen)
     (at tray134 kitchen)
     (at_kitchen_bread bread464)
     (at_kitchen_bread bread40)
     (at_kitchen_content content463)
     (at_kitchen_content content188)
     (not_allergic_gluten child489)
     (not_allergic_gluten child3)
     (waiting child3 table118)
     (waiting child489 table69)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child3)
     (served child489)
    )
  )
)
