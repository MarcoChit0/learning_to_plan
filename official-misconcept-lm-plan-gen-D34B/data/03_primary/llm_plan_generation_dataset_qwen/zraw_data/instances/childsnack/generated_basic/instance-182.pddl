; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 485039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 child316 - child
    bread429 bread209 - bread-portion
    content207 content202 - content-portion
    tray325 tray327 tray449 - tray
    table91 table165 table430 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray325 kitchen)
     (at tray327 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread429)
     (at_kitchen_bread bread209)
     (at_kitchen_content content207)
     (at_kitchen_content content202)
     (not_allergic_gluten child97)
     (not_allergic_gluten child316)
     (waiting child97 table430)
     (waiting child316 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child97)
     (served child316)
    )
  )
)
