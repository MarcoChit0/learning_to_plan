; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 352939

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child418 child463 - child
    bread107 bread27 - bread-portion
    content180 content201 - content-portion
    tray36 tray361 - tray
    table151 table374 table476 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread107)
     (at_kitchen_bread bread27)
     (at_kitchen_content content180)
     (at_kitchen_content content201)
     (no_gluten_bread bread107)
     (no_gluten_content content180)
     (allergic_gluten child463)
     (not_allergic_gluten child418)
     (waiting child418 table151)
     (waiting child463 table374)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child418)
     (served child463)
    )
  )
)
