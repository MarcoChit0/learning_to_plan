; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 789143

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 child375 - child
    bread17 bread119 - bread-portion
    content316 content6 - content-portion
    tray107 tray70 tray164 tray266 - tray
    table374 table34 table94 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray107 kitchen)
     (at tray70 kitchen)
     (at tray164 kitchen)
     (at tray266 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_bread bread119)
     (at_kitchen_content content316)
     (at_kitchen_content content6)
     (not_allergic_gluten child375)
     (not_allergic_gluten child31)
     (waiting child31 table34)
     (waiting child375 table374)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child31)
     (served child375)
    )
  )
)
