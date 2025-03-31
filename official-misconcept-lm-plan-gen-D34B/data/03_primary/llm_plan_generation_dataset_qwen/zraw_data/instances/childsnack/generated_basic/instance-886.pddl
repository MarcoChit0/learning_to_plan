; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 732087

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child350 child140 - child
    bread452 bread108 bread184 - bread-portion
    content319 content301 content77 - content-portion
    tray36 - tray
    table39 table481 table433 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread108)
     (at_kitchen_bread bread184)
     (at_kitchen_content content319)
     (at_kitchen_content content301)
     (at_kitchen_content content77)
     (not_allergic_gluten child16)
     (not_allergic_gluten child140)
     (not_allergic_gluten child350)
     (waiting child16 table433)
     (waiting child350 table39)
     (waiting child140 table433)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child16)
     (served child350)
     (served child140)
    )
  )
)
