; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 921402

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child37 - child
    bread394 bread131 - bread-portion
    content363 content54 - content-portion
    tray198 tray184 - tray
    table246 table108 table304 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray184 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread131)
     (at_kitchen_content content363)
     (at_kitchen_content content54)
     (no_gluten_bread bread394)
     (no_gluten_content content54)
     (allergic_gluten child85)
     (not_allergic_gluten child37)
     (waiting child85 table108)
     (waiting child37 table304)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child85)
     (served child37)
    )
  )
)
