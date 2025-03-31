; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 775205

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 child37 - child
    bread71 bread251 - bread-portion
    content450 content418 - content-portion
    tray224 tray271 - tray
    table28 table491 table245 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray224 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_bread bread251)
     (at_kitchen_content content450)
     (at_kitchen_content content418)
     (no_gluten_bread bread71)
     (no_gluten_content content418)
     (allergic_gluten child436)
     (not_allergic_gluten child37)
     (waiting child436 table491)
     (waiting child37 table491)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child436)
     (served child37)
    )
  )
)
