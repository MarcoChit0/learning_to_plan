; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 421714

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 child87 - child
    bread136 bread462 - bread-portion
    content9 content231 - content-portion
    tray337 tray100 tray383 - tray
    table470 table157 table117 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at tray100 kitchen)
     (at tray383 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread462)
     (at_kitchen_content content9)
     (at_kitchen_content content231)
     (no_gluten_bread bread136)
     (no_gluten_content content9)
     (allergic_gluten child87)
     (not_allergic_gluten child196)
     (waiting child196 table470)
     (waiting child87 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child196)
     (served child87)
    )
  )
)
