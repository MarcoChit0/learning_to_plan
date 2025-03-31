; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 977981

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 child246 - child
    bread466 bread136 - bread-portion
    content88 content108 - content-portion
    tray49 tray451 - tray
    table61 table443 table224 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray451 kitchen)
     (at_kitchen_bread bread466)
     (at_kitchen_bread bread136)
     (at_kitchen_content content88)
     (at_kitchen_content content108)
     (no_gluten_bread bread136)
     (no_gluten_content content108)
     (allergic_gluten child170)
     (not_allergic_gluten child246)
     (waiting child170 table443)
     (waiting child246 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child170)
     (served child246)
    )
  )
)
