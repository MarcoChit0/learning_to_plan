; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 2085

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child195 - child
    bread428 bread62 - bread-portion
    content189 content158 - content-portion
    tray302 tray104 tray261 - tray
    table274 table8 table124 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray104 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread62)
     (at_kitchen_content content189)
     (at_kitchen_content content158)
     (not_allergic_gluten child195)
     (not_allergic_gluten child51)
     (waiting child51 table274)
     (waiting child195 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child51)
     (served child195)
    )
  )
)
