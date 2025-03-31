; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 756037

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 child440 - child
    bread94 bread388 - bread-portion
    content103 content35 - content-portion
    tray496 tray104 tray122 - tray
    table180 table220 table205 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray496 kitchen)
     (at tray104 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread388)
     (at_kitchen_content content103)
     (at_kitchen_content content35)
     (not_allergic_gluten child272)
     (not_allergic_gluten child440)
     (waiting child272 table180)
     (waiting child440 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child272)
     (served child440)
    )
  )
)
